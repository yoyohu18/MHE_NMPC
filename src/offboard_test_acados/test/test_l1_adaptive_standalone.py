#!/usr/bin/env python3
# C.1 Phase 0 单测(跟 test_mhe_standalone.py 同一个套路):不需要 ROS2/PX4/acados,
# 纯 numpy 验证 L1Augmentation 的估计律本身对不对。真值系统独立实现(避免测试
# 正确性依赖被测代码同一份公式)。运行:
#   python3 test_l1_adaptive_standalone.py   (或 pytest)
#
# 覆盖(对应实施计划 §4 落点 6 的三条 + 闭环概念验证):
#   1 零扰动零漂移              4 LPF 噪声抑制(omega_c 的鲁棒作用)
#   2 常值扰动收敛到真值        5 三通道独立
#   3 阶跃跟踪且入带时间随 omega_c 单调  6 reset 语义(drop 场景)
#   7 闭环概念验证:1-D 悬停 + 质量阶跃,L1 补偿 vs 无补偿的高度误差

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from offboard_test_acados.l1_adaptive import L1Augmentation  # noqa: E402

np.random.seed(42)
DT = 0.02          # 50Hz,对齐 NMPC 控制周期量级
G = 9.81


def _run_open_loop(l1, d_true_fn, T=6.0, noise=0.0):
    """开环校验台:真值系统 v̇ = a_nom + d_true(t),a_nom 恒零(纯看扰动通道)。
    返回 (t 数组, d̂_f 历史)。"""
    n = int(T / DT)
    v = np.zeros(l1.dim)
    hist = np.zeros((n, l1.dim))
    t = np.arange(n) * DT
    for k in range(n):
        d = np.asarray(d_true_fn(t[k]), dtype=float)
        v = v + DT * d                       # 真值传播(a_nom=0)
        v_meas = v + noise * np.random.randn(l1.dim)
        hist[k] = l1.update(v_meas, np.zeros(l1.dim), DT)
    return t, hist


def test_zero_disturbance_no_drift():
    l1 = L1Augmentation()
    _, hist = _run_open_loop(l1, lambda t: np.zeros(3), T=4.0)
    assert np.max(np.abs(hist)) < 1e-9, '零扰动下 d̂_f 必须恒零(无漂移)'


def test_constant_disturbance_convergence():
    d_true = np.array([-1.4, 0.6, -2.9])     # 量级≈0.3kg 载荷的比力缺口
    l1 = L1Augmentation()
    _, hist = _run_open_loop(l1, lambda t: d_true, T=6.0)
    err = np.abs(hist[-1] - d_true) / np.abs(d_true)
    assert np.all(err < 0.02), f'常值扰动稳态误差应<2%,实际 {err}'


def test_step_tracking_monotone_in_omega_c():
    """attach 等效:t=1s 扰动 0→-1.4。入带(90% 幅值)时间应随 omega_c 增大而缩短。"""
    def d_fn(t):
        return np.array([-1.4 if t >= 1.0 else 0.0, 0.0, 0.0])
    t_enter = []
    for wc in (2.0, 5.0, 10.0):
        l1 = L1Augmentation(omega_c=wc)
        t, hist = _run_open_loop(l1, d_fn, T=5.0)
        inband = (t >= 1.0) & (hist[:, 0] < -1.4 * 0.9)
        assert np.any(inband), f'omega_c={wc} 阶跃后未入带'
        t_enter.append(t[np.argmax(inband)] - 1.0)
    assert t_enter[0] > t_enter[1] > t_enter[2], \
        f'入带时间应随 omega_c 单调缩短,实际 {t_enter}'
    # 阶跃前不得有预响应
    assert np.max(np.abs(hist[t < 1.0])) < 1e-9


def test_lpf_noise_rejection():
    """同一测量噪声下,omega_c 小 → d̂_f 稳态方差应显著更小(鲁棒旋钮有效)。"""
    d_true = np.array([-1.4, 0.0, 0.0])
    stds = []
    for wc in (2.0, 20.0):
        np.random.seed(7)                    # 两组同噪声序列(CRN)
        l1 = L1Augmentation(omega_c=wc)
        _, hist = _run_open_loop(l1, lambda t: d_true, T=8.0, noise=0.02)
        stds.append(np.std(hist[-150:, 0]))  # 稳态段
    assert stds[0] < 0.3 * stds[1], \
        f'omega_c=2 的稳态噪声应远小于 omega_c=20,实际 {stds}'


def test_channel_independence():
    """只激励 y 通道,x/z 的 d̂_f 必须保持零(A_s 对角,无通道串扰)。"""
    l1 = L1Augmentation()
    _, hist = _run_open_loop(l1, lambda t: np.array([0.0, -2.0, 0.0]), T=4.0)
    assert abs(hist[-1, 1] + 2.0) < 0.05
    assert np.max(np.abs(hist[:, [0, 2]])) < 1e-9


def test_reset_semantics():
    """drop 场景:收敛后 reset,输出必须立刻归零且不残留(残留会拽偏推力)。"""
    d_true = np.array([-1.4, 0.0, 0.0])
    l1 = L1Augmentation()
    _run_open_loop(l1, lambda t: d_true, T=4.0)
    assert abs(l1.d_hat_f[0] + 1.4) < 0.05   # 先确认确实收敛了
    l1.reset()
    assert np.all(l1.d_hat_f == 0.0) and np.all(l1.d_hat == 0.0)
    out = l1.update(np.array([0.3, 0.0, 0.0]), np.zeros(3), DT)
    assert np.all(out == 0.0), '复位后首帧应为零(预测器重新对齐)'


def test_closed_loop_mass_step_hover():
    """闭环概念验证(Phase 0 的物理证明,1-D 垂直通道):
    悬停 PD + 标称质量前馈,t=2s 质量 2.064→2.364(attach 0.3kg)。控制器不知道
    新质量(流派 B 设定);对照=无补偿,实验=推力叠加 m_nom·d̂_f。判据:L1 使
    突变后的稳态高度误差被压掉 >90%(无补偿时 PD 有恒定稳态误差)。"""
    m_nom, m_new = 2.064, 2.364
    kp, kd_ = 8.0, 4.0
    z_ref = 3.0

    def sim(use_l1):
        l1 = L1Augmentation(a_gain=10.0, omega_c=5.0, dim=1)
        z, vz = z_ref, 0.0
        n = int(8.0 / DT)
        z_hist = np.zeros(n)
        for k in range(n):
            m = m_new if k * DT >= 2.0 else m_nom
            d_f = l1.d_hat_f[0] if use_l1 else 0.0
            # 标称悬停前馈 + PD + L1 补偿(比力→推力乘标称质量)
            thrust = m_nom * (G + kp * (z_ref - z) - kd_ * vz - d_f)
            acc = thrust / m - G                 # 真实动力学(真实质量)
            a_nom = thrust / m_nom - G           # 标称模型预测(标称质量)
            l1.update(np.array([vz]), np.array([a_nom]), DT)
            vz += DT * acc
            z += DT * vz
            z_hist[k] = z
        return z_hist

    z_off = sim(False)
    z_on = sim(True)
    e_off = abs(z_off[-1] - z_ref)               # 无补偿稳态误差(应明显非零)
    e_on = abs(z_on[-1] - z_ref)
    assert e_off > 0.10, f'对照组应有明显稳态误差,实际 {e_off:.3f}'
    assert e_on < 0.1 * e_off, \
        f'L1 应压掉>90% 稳态误差: off={e_off:.3f} on={e_on:.4f}'


if __name__ == '__main__':
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print(f'PASS  {name}')
            except AssertionError as e:
                fails += 1
                print(f'FAIL  {name}: {e}')
    print('=' * 50)
    print('全部通过' if fails == 0 else f'{fails} 个失败')
    sys.exit(1 if fails else 0)
