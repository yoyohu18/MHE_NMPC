#!/usr/bin/env python3
# LIFT 中途 hold + dJ 跟随 m_est 的验证(2026-08-25)。不需要 acados/ROS。
#
# 背景(gviz_20260825_201344 实测):attach 后 box 仍在地上时 T_phys 恒为空机重量,
# m_est 反而从 2.064 下漂到 2.023 —— 延长 attach→离地窗口对 MHE 毫无用处。
# 离地后 0.6s m_est 就到 2.319(真值 2.364)。但 dJ_est 在 attach 时按 prior 定死
# 后再不更新,那 1.9s 的有效信息被浪费,随后 LIFT 段发散。
# 本模块测两件事:①hold 把斜坡冻在离地之后 ②dJ 棘轮把 m_est 的增长送出去。

import numpy as np

from offboard_test_acados import acados_nmpc_node as an
from offboard_test_acados.acados_params import p


class _Stub:
    """最小 stub;方法本体从真实类上借,测的是真实实现。"""
    def __init__(self, **kw):
        self.gripper_mode = True
        self.attach_time = 7.7
        self.grip_lift_after_sec = 1.5
        self.grip_lift_dur = 8.4
        self.grip_z_low, self.grip_z_high = 0.55, 6.0
        self.grip_lift_started = False
        self.z_hover = self.grip_z_low
        self.grip_lift_hold_enable = True
        self.grip_lift_hold_dz = 0.35
        self.grip_lift_hold_sec = 3.0
        self._lift_hold_start = None
        self._lift_frozen_sec = 0.0
        self._lift_hold_done = False
        self.dj_track_mest = True
        self.grip_mp_cap = 0.6
        self.grip_dj_floor_mp = 0.15
        self.dj_ratchet_enable = True
        self.grip_payload_envelope = 0.5
        self.dj_gain_rescale_dmp = 0.05
        self.dj_gain_rescale_ratio = 1.25
        self._mp_ratchet = 0.0
        # attach(online 分支)之后的真实状态:棘轮从 0 起步(模型侧要能从包线
        # 过估精修下来),增益滞回基准已对齐到包线(执行器侧只许上调)。
        self._mp_gain_applied = 0.5
        self._geom_log_count = 0
        self.grip_mass_stepped = True
        self.grip_dropped = False
        self.grip_arm_d = 0.47
        self.m_est = p.m
        self.dJ_est = 0.0
        self.c_est = np.zeros(2)
        self.c_xy_online = np.zeros(2)
        self.gain_calls = []
        self.__dict__.update(kw)

    def get_logger(self):
        class _L:
            info = warn = staticmethod(lambda m: None)
        return _L()

    def _scale_px4_rate_gains(self, dJ, allow_rescale=False):
        self.gain_calls.append(dJ)


_lift = an.AcadosNMPCNode._lift_phase
_geom = an.AcadosNMPCNode._update_online_geometry
_dJ = an.AcadosNMPCNode._dJ_from_mp
_Stub._dJ_from_mp = _dJ


def test_hold_freezes_ramp():
    """hold 期间 z_hover 必须原地不动,hold 结束后继续爬。"""
    s = _Stub()
    t0 = s.attach_time + s.grip_lift_after_sec
    z_at = {}
    for k in range(int(20.0 / 0.05)):
        t = t0 + k * 0.05
        _lift(s, t)
        z_at[round(t, 2)] = s.z_hover
    assert s._lift_hold_start is not None, '应该进入过 hold'
    assert s._lift_hold_done, '应该完成 hold'
    th = s._lift_hold_start
    z_in = [z for tt, z in z_at.items() if th + 0.1 <= tt <= th + s.grip_lift_hold_sec - 0.1]
    assert max(z_in) - min(z_in) < 1e-9, f'hold 期间 z 不该变,实测跨度 {max(z_in)-min(z_in):.2e}'
    z_after = [z for tt, z in z_at.items() if tt > th + s.grip_lift_hold_sec + 0.5]
    assert z_after[-1] > max(z_in) + 0.1, 'hold 结束后应继续爬升'
    print(f'[1] hold 冻结 OK (在 dz={max(z_in)-s.grip_z_low:.3f}m 处冻 '
          f'{s.grip_lift_hold_sec}s, 之后爬到 {z_after[-1]:.2f}m)')


def test_hold_triggers_after_liftoff_height():
    """hold 必须发生在抬升 >= hold_dz 之后(即 box 已离地),不能在 attach 处。"""
    s = _Stub()
    t0 = s.attach_time + s.grip_lift_after_sec
    for k in range(400):
        _lift(s, t0 + k * 0.05)
        if s._lift_hold_start is not None:
            break
    dz = s.z_hover - s.grip_z_low
    assert dz >= s.grip_lift_hold_dz - 1e-6, f'hold 触发时 dz={dz:.3f} < {s.grip_lift_hold_dz}'
    assert dz < s.grip_lift_hold_dz + 0.15, f'hold 触发过晚 dz={dz:.3f}'
    print(f'[2] hold 触发点 OK (dz={dz:.3f}m ≥ {s.grip_lift_hold_dz}m)')


def test_dj_ratchet_monotonic_and_capped():
    """dJ 只增不减(治 LIFT 段下漂冻结),且封顶(治高估无界跟涨)。"""
    s = _Stub()
    # 复刻实测序列:attach 后下漂 -> 离地爬升 -> 抖动回落 -> 异常高估
    seq = [2.064, 2.037, 2.023, 2.068, 2.319, 2.329, 2.297, 2.249, 2.356,
           2.20, 3.734]
    dJs = []
    for m in seq:
        s.m_est = m
        _geom(s)
        dJs.append(s.dJ_est)
    assert all(b >= a - 1e-12 for a, b in zip(dJs, dJs[1:])), f'dJ 非单调: {dJs}'
    # 下漂段(前三个)棘轮不动,模型 dJ 停在地板上——不是 0(dJ→0 会让 NMPC
    # 按空机惯量规划角加速度,力矩饱和后级联发散,见 grip_dj_floor_mp 注释)
    dJ_floor = _dJ(s, s.grip_dj_floor_mp)
    assert abs(dJs[2] - dJ_floor) < 1e-12, \
        f'box 在地上时 dJ 应停在地板 {dJ_floor:.4f},实得 {dJs[2]:.4f}'
    # 回落不回退
    assert dJs[9] == dJs[8], 'm_est 回落时 dJ 不该跟着降'
    # 高估被 cap 住
    dJ_cap = _dJ(s, s.grip_mp_cap)
    assert abs(dJs[-1] - dJ_cap) < 1e-12, f'高估应被封顶到 {dJ_cap:.4f},实得 {dJs[-1]:.4f}'
    dJ_true = _dJ(s, 0.3)
    print(f'[3] dJ 棘轮 OK (地上=地板 {dJs[2]:.4f} → 离地 {dJs[4]:.4f} → 真值参考 '
          f'{dJ_true:.4f} → 高估封顶 {dJs[-1]:.4f})')


def test_gain_not_dragged_down_by_ratchet():
    """★ 内环增益不许被棘轮往下拖。attach 时已按包线上界(0.5kg)缩放过,棘轮
    从 0 爬到真值 0.3 全程都在包线之下,不该产生任何重缩放调用。

    不设这条保护时的实测代数:_mp_gain_applied 停在 0 → 棘轮第一步 m_p≈0.05
    就满足 >= dj_gain_rescale_dmp=0.05 → 增益比从包线的 5.00 掉到 1.76(35%),
    而 attach 那次缩放的存在理由正是"不等估计收敛,否则实测炸机"。"""
    s = _Stub()
    for m in np.linspace(2.064, 2.364, 60):   # 平滑爬升 60 帧,m_p: 0 → 0.30
        s.m_est = float(m)
        _geom(s)
    assert s._mp_ratchet > 0.29, f'棘轮应爬到真值附近,实得 {s._mp_ratchet:.3f}'
    assert s.gain_calls == [], \
        f'棘轮在包线之下时不该动增益,实得 {len(s.gain_calls)} 次: {s.gain_calls}'
    print(f'[4] 增益不被棘轮下拖 OK (棘轮爬到 {s._mp_ratchet:.3f} < 包线 '
          f'{s.grip_payload_envelope}, 0 次重缩放)')


def test_gain_rescale_unreachable_at_default_params():
    """⚠️ 记录一个参数间的既有关系:**默认参数下增益上调这条路不可达**。
    滞回门槛 = max(dj_gain_rescale_dmp, _mp_gain_applied*(ratio-1))
             = max(0.05, 0.5*0.25) = 0.125 → 棘轮要到 0.625 才触发,
    而 grip_mp_cap=0.6 先把棘轮封死。两个数都表达"机架规格上界",所以自洽:
    默认配置下增益就钉在 attach 时的包线整定上,永不改动。
    要让上调真的能发生,得抬 grip_mp_cap 或降 dj_gain_rescale_ratio。"""
    s = _Stub()
    for m in np.linspace(2.064, 3.2, 200):    # m_p 一路顶到 cap 0.6
        s.m_est = float(m)
        _geom(s)
    assert abs(s._mp_ratchet - s.grip_mp_cap) < 1e-12, '棘轮应被 cap 封顶'
    assert s.gain_calls == [], \
        f'默认参数下不该有任何上调(门槛 0.625 > cap 0.6),实得 {s.gain_calls}'
    print(f'[5a] 默认参数下增益上调不可达 OK (棘轮封顶 {s._mp_ratchet:.2f}, '
          f'门槛 0.625)')


def test_gain_still_rescales_above_envelope():
    """但机制本身要在:棘轮真的涨过包线(机架规格给小了)时必须能往上调,
    且要有滞回——每抖一下就重设 PX4 参数会刷服务调用,而且 PX4 会把它当真机
    参数落盘。用放宽的 cap 测机制,不测默认取值(见上一条)。"""
    s = _Stub(grip_mp_cap=1.2)
    for m in np.linspace(2.064, 3.164, 200):  # m_p: 0 → 1.10,越过门槛 0.625
        s.m_est = float(m)
        _geom(s)
    assert len(s.gain_calls) >= 1, '棘轮越过包线后应至少上调一次'
    assert len(s.gain_calls) <= 4, f'滞回失效,实得 {len(s.gain_calls)} 次'
    assert all(b >= a for a, b in zip(s.gain_calls, s.gain_calls[1:])), \
        f'增益只该单调上调: {s.gain_calls}'
    assert s.gain_calls[0] >= _dJ(s, s.grip_payload_envelope), \
        '第一次上调时棘轮就该已经超过包线'
    print(f'[5b] 越过包线才上调 OK ({len(s.gain_calls)} 次: '
          f'{[round(v, 4) for v in s.gain_calls]})')


def test_floor_zero_restores_bare_ratchet():
    """grip_dj_floor_mp=0 = 无地板,退回裸棘轮语义(给要复现旧行为的批次留后路)。"""
    s = _Stub(grip_dj_floor_mp=0.0)
    s.m_est = p.m - 0.04          # box 还在地上,m_est 下漂到空机以下
    _geom(s)
    assert s.dJ_est == 0.0, f'无地板时 dJ 应为 0,实得 {s.dJ_est}'
    print('[6] floor=0 退回裸棘轮 OK')


def test_no_ratchet_tracks_both_ways():
    """dj_ratchet_enable=False:模型 dJ 双向跟随 m_est,不再记住峰值。
    地板和 cap 两个静态界仍然生效(它们才是治高估/治 LIFT 低值的那两道),
    而增益侧**照旧**用棘轮峰值——两者由不同语义驱动,不该被同一个开关连坐。"""
    s = _Stub(dj_ratchet_enable=False)
    # 涨 → 更涨 → 回落 → 再涨。回落值取 2.25(m_p=0.186)而不是 2.20
    # (m_p=0.136)——后者低于地板 0.15,会被地板抬住,测不出"跟着降"。
    seq = [2.30, 2.40, 2.25, 2.35]
    dJs = []
    for m in seq:
        s.m_est = m
        _geom(s)
        dJs.append(s.dJ_est)
    assert dJs[2] < dJs[1], f'回落时 dJ 应跟着降(无棘轮),实得 {dJs}'
    assert abs(dJs[2] - _dJ(s, 2.25 - p.m)) < 1e-12, 'dJ 应等于当前 m_p 对应值'
    # 棘轮本身仍在走(增益侧要用),只是模型侧不再消费它
    assert abs(s._mp_ratchet - (2.40 - p.m)) < 1e-9, \
        f'棘轮仍应记住峰值供增益侧使用,实得 {s._mp_ratchet:.3f}'
    # 地板仍然兜底
    s.m_est = p.m - 0.05                  # 探到空机以下
    _geom(s)
    assert abs(s.dJ_est - _dJ(s, s.grip_dj_floor_mp)) < 1e-12, \
        '无棘轮时地板仍须兜住 dJ'
    print(f'[9] 无棘轮双向跟随 OK (dJ {[round(v, 4) for v in dJs]}, '
          f'棘轮仍走到 {s._mp_ratchet:.3f}, 探底后回到地板 {s.dJ_est:.4f})')


def test_drop_releases_ratchet():
    """drop 后棘轮必须一起卸掉,否则空机还挂着幽灵惯量 + 过增益。"""
    s = _Stub(m_est=2.364)
    _geom(s)
    assert s._mp_ratchet > 0
    s.grip_dropped = True
    _geom(s)
    assert s.dJ_est == 0.0 and s._mp_ratchet == 0.0, 'drop 后应清零'
    print('[7] drop 释放棘轮 OK')


def test_disabled_keeps_legacy_behaviour():
    """开关关闭时行为逐字节回到 legacy:只更新 c_est,dJ 一动不动。"""
    s = _Stub(dj_track_mest=False, dJ_est=0.0588, m_est=3.0)
    s.c_xy_online = np.array([0.01, -0.02])
    _geom(s)
    assert s.dJ_est == 0.0588, 'legacy 下 dJ 不该被改'
    assert np.allclose(s.c_est, [0.01, -0.02]), 'c_est 仍应更新'
    assert s.gain_calls == [], 'legacy 下不该动增益'
    print('[8] 开关关闭 = legacy 行为 OK')


if __name__ == '__main__':
    test_hold_freezes_ramp()
    test_hold_triggers_after_liftoff_height()
    test_dj_ratchet_monotonic_and_capped()
    test_gain_not_dragged_down_by_ratchet()
    test_gain_rescale_unreachable_at_default_params()
    test_gain_still_rescales_above_envelope()
    test_floor_zero_restores_bare_ratchet()
    test_no_ratchet_tracks_both_ways()
    test_drop_releases_ratchet()
    test_disabled_keeps_legacy_behaviour()
    print('\nall passed')
