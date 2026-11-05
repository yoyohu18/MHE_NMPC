#!/usr/bin/env python3
"""动力学残差**可辨识性诊断**的原始流采集器(2026-07-31)。

【这是诊断采集,不是"神经网络训练数据采集"】目的先于建模:回答残差是否
①明显大于标定与导数不确定度 ②能被当前可观测状态预测 ③跨轮次可重复
④跨质量/偏心/轨迹泛化。四条不全过就不该上网络。

【为什么分四个异步流而不在线拼成一张表】motor / odom / command 三个话题频率
与时钟来源都不同(motor 经 ros_gz_bridge 来自 Gazebo,odom 经 mavros 来自 PX4,
u_opt 是本机 NMPC 直发)。若在某个回调里把三者凑成一行,行内字段就不是同一
采样时刻,而这个错位**恰好会被误当成模型残差**——正是要诊断的东西被采集方式
污染。故每条消息各写各的流,时间对齐留给离线后处理(可自由做延迟扫描/插值/
最近邻/重采样/丢包检测)。

【两套时间戳】
  t_header_ns : 消息自带 header.stamp,用于物理时间对齐
  t_recv_ns   : 回调收到时的**单调钟**,用于诊断传输抖动
两者之差是否稳定 = 该流延迟是否稳定。⚠️三个流的 header.stamp 是否处于同一
时钟域**必须用数据验证**(本工程未设 use_sim_time,而 motor/odom 分别来自
Gazebo 与 PX4),不能因为字段都叫 header.stamp 就直接相减。

【command 流没有 header】/acados_nmpc/u_opt 是 Float64MultiArray,**不带时间戳**,
只能记接收时刻。且它是 **NMPC 的理想优化输出**,之后还要被转成 body-rate
setpoint 交给 PX4 内环——它**不是执行器输入**,不能直接拿来和 tau_phys 做执行器
延迟扫描。真正的执行器输出是 motor 流反算的 tau_phys。

用法:给 mhe_node 传 -p residual_log_dir:=<目录>(默认空=完全关闭,零开销)。
"""
import time
from pathlib import Path

_SCHEMA = {
    'motor':   't_header_ns,t_recv_ns,w1,w2,w3,w4,tau_x,tau_y,tau_z,T_phys',
    'odom':    ('t_header_ns,t_recv_ns,px,py,pz,qw,qx,qy,qz,'
                'vb_x,vb_y,vb_z,wb_x,wb_y,wb_z'),
    # IMU 独立成流:odom 实测只有 ~31Hz,拿它差分算 omega_dot 会让残差被微分噪声
    # 主导(而 motor 端是 250Hz)。IMU 的 angular_velocity 通常频率高得多,是算
    # omega_dot 的更好来源。两个源都采,离线对比后再决定用哪个,不在线预判。
    'imu':     ('t_header_ns,t_recv_ns,wx,wy,wz,ax,ay,az,qw,qx,qy,qz'),
    'command': 't_recv_ns,T_cmd,tau_x,tau_y,tau_z',
    # 2026-08-24 追加真值对照列(**只在末尾追加**,消费者都用 genfromtxt(names=True)
    # 按列名取,老脚本不受影响)。m_true/c_*_true/dJ_true/J*_true 全部只从
    # eval_true_payload_mass 这个**评估专用**参数算,绝不进 MHE/NMPC 的任何模型
    # 输入(见 mhe_node 里该参数的注释);没给真值时写 nan。
    'internal': ('t_recv_ns,m_est,payload_attached,'
                 'att_off_x,att_off_y,att_off_z,thrust_phys,'
                 'm_true,cx_hat,cy_hat,cx_true,cy_true,'
                 'dJ_hat,dJ_true,Jxx_hat,Jxx_true,Jyz_hat,Jyz_true'),
}
# 攒够多少行写一次 + 立刻 flush。
# ⚠️2026-07-31 首次采集全军覆没的教训:原来用 open(buffering=1<<20) 给了 1MB 文件
# 缓冲,`write()` 只是进 Python 缓冲、**没落盘**;批次每轮用 kill -9 收尾,缓冲连同
# 表头一起蒸发,四条流里三条 0 字节。修法 = 每批 write 后显式 flush()。
# 只需 flush() 不需 fsync():flush 后数据已交给内核 page cache,进程被 kill -9 也不
# 丢(只有掉电才丢),而 fsync 每 0.2s 一次会拖垮高频回调。
_FLUSH_EVERY = 50


class ResidualLogger:
    """四条异步原始流 → 四个 CSV。默认不启用;启用时对回调只加一次 list.append。"""

    def __init__(self, out_dir, stamp, meta=None):
        self.enabled = bool(out_dir)
        if not self.enabled:
            return
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.stamp = stamp
        self._buf = {k: [] for k in _SCHEMA}
        self._fh = {}
        for name, header in _SCHEMA.items():
            p = self.dir / f'resid_{name}_{stamp}.csv'
            fh = open(p, 'w', buffering=1 << 16)
            fh.write(header + '\n')
            fh.flush()                 # 表头立刻落盘,别等第一批数据
            self._fh[name] = fh
        if meta:
            self._write_manifest(meta)

    def _write_manifest(self, meta):
        """标定常数/惯量/真值条件/坐标系语义全部落盘——残差诊断的结论完全依赖
        这些数,事后无法回忆。"""
        with open(self.dir / f'resid_manifest_{self.stamp}.txt', 'w') as f:
            f.write(f'# 残差诊断采集 manifest {self.stamp}\n')
            for k, v in meta.items():
                f.write(f'{k} = {v}\n')

    @staticmethod
    def _hdr_ns(msg):
        try:
            s = msg.header.stamp
            return s.sec * 1_000_000_000 + s.nanosec
        except AttributeError:
            return -1                      # 该消息类型不带 header

    def _put(self, name, row):
        buf = self._buf[name]
        buf.append(row)
        if len(buf) >= _FLUSH_EVERY:
            fh = self._fh[name]
            fh.write('\n'.join(buf) + '\n')
            fh.flush()                 # 见 _FLUSH_EVERY 注释:不 flush 就等于没写
            buf.clear()

    # ---- 四条流各自的入口(在对应回调里调一次) ----
    def log_motor(self, msg, w, tau, thrust):
        if not self.enabled:
            return
        self._put('motor', f'{self._hdr_ns(msg)},{time.monotonic_ns()},'
                           f'{w[0]:.6f},{w[1]:.6f},{w[2]:.6f},{w[3]:.6f},'
                           f'{tau[0]:.8f},{tau[1]:.8f},{tau[2]:.8f},{thrust:.6f}')

    def log_odom(self, msg):
        if not self.enabled:
            return
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        v, w = msg.twist.twist.linear, msg.twist.twist.angular
        # v/w 记**原始 body(FLU)** 值,不记 odom_cb 转换后的 world 速度——诊断要
        # 的是未经我方变换的原始观测。
        self._put('odom', f'{self._hdr_ns(msg)},{time.monotonic_ns()},'
                          f'{p.x:.6f},{p.y:.6f},{p.z:.6f},'
                          f'{q.w:.8f},{q.x:.8f},{q.y:.8f},{q.z:.8f},'
                          f'{v.x:.6f},{v.y:.6f},{v.z:.6f},'
                          f'{w.x:.6f},{w.y:.6f},{w.z:.6f}')

    def log_imu(self, msg):
        if not self.enabled:
            return
        w, a, q = msg.angular_velocity, msg.linear_acceleration, msg.orientation
        self._put('imu', f'{self._hdr_ns(msg)},{time.monotonic_ns()},'
                         f'{w.x:.8f},{w.y:.8f},{w.z:.8f},'
                         f'{a.x:.6f},{a.y:.6f},{a.z:.6f},'
                         f'{q.w:.8f},{q.x:.8f},{q.y:.8f},{q.z:.8f}')

    def log_command(self, u):
        if not self.enabled:
            return
        self._put('command', f'{time.monotonic_ns()},'
                             f'{u[0]:.6f},{u[1]:.8f},{u[2]:.8f},{u[3]:.8f}')

    def log_internal(self, m_est, attached, att_off, thrust, truth=None):
        """truth: 可选 dict,键见 _SCHEMA['internal'] 追加段的列名(不给就写 nan)。"""
        if not self.enabled:
            return
        a = att_off if att_off is not None else (float('nan'),) * 3
        t = thrust if thrust is not None else float('nan')
        tr = truth or {}
        cols = ('m_true', 'cx_hat', 'cy_hat', 'cx_true', 'cy_true',
                'dJ_hat', 'dJ_true', 'Jxx_hat', 'Jxx_true',
                'Jyz_hat', 'Jyz_true')
        extra = ','.join(f'{float(tr.get(k, float("nan"))):.6f}' for k in cols)
        self._put('internal', f'{time.monotonic_ns()},{m_est:.6f},'
                              f'{int(bool(attached))},'
                              f'{a[0]:.6f},{a[1]:.6f},{a[2]:.6f},{t:.6f},'
                              f'{extra}')

    def close(self):
        if not self.enabled:
            return
        for name, buf in self._buf.items():
            if buf:
                self._fh[name].write('\n'.join(buf) + '\n')
            self._fh[name].close()
