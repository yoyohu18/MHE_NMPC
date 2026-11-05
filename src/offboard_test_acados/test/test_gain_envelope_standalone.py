#!/usr/bin/env python3
# #2「去先验」方案 b:用载荷**包线上界**代替**点估计先验**驱动 PX4 内环增益。
# 核心论证是纯数学的(ratio 有 cap 5.0),所以先在这里锁死,不必烧 SITL 机时:
#   dJ(m_p) = [m_B·m_p/(m_B+m_p)]·d²,   ratio = min((Jxx+dJ)/Jxx, 5.0)
# → 存在临界载荷 m_p*,凡 m_p >= m_p* 的**任何**取值给出**同一个**增益。
# 于是"那个先验的数值精度从未被使用,只有'够不够大'被使用",而"够大"正是
# 包线的语义 —— 这就是替换在 cap 内免费的理由。
#
# 运行: python3 -m pytest test/test_gain_envelope_standalone.py -q

import numpy as np
from offboard_test_acados.acados_params import p

ARM_D = 0.47          # grip_arm_d 默认值(与 acados_nmpc_node 参数声明一致)
GAIN_CAP = 5.0        # _scale_px4_rate_gains 里的 ratio 上限


def dJ_from_prior(m_p, arm_d=ARM_D):
    """复刻 acados_nmpc_node._dJ_from_prior。"""
    m_t = p.m + m_p
    mu = p.m * m_p / m_t if m_t > 0 else 0.0
    return float(mu * arm_d ** 2)


def ratio(m_p, J0=None):
    """复刻 _scale_px4_rate_gains 的增益比(含 cap)。"""
    J0 = p.Jxx if J0 is None else J0
    return min((J0 + dJ_from_prior(m_p)) / J0, GAIN_CAP)


def critical_mass(arm_d=ARM_D, J0=None):
    """解析解:ratio 恰好撞 cap 的载荷质量 m_p*。"""
    J0 = p.Jxx if J0 is None else J0
    mu = (GAIN_CAP - 1.0) * J0 / arm_d ** 2
    return mu * p.m / (p.m - mu)


def test_critical_mass_matches_numeric():
    """解析临界质量与数值扫描一致(误差 <1mg)。"""
    m_star = critical_mass()
    grid = np.linspace(0.05, 1.0, 200001)
    first = grid[[ratio(x) >= GAIN_CAP - 1e-12 for x in grid]][0]
    assert abs(first - m_star) < 1e-3, (first, m_star)
    assert 0.29 < m_star < 0.30, m_star      # 记在注释里的 0.2937kg


def test_envelope_equals_point_prior_above_cap():
    """主线工况 m_p=0.3:包线上界(0.4/0.5/1.0)与点估计 0.3 给出逐位相同的增益。"""
    assert ratio(0.3) == GAIN_CAP
    for env in (0.4, 0.5, 1.0, 3.0):
        assert ratio(env) == ratio(0.3), env


def test_envelope_differs_below_cap():
    """轻载格子不在 cap 里:换包线是真的改整定,不能宣称"逐位不变"。"""
    for m_p, expect in ((0.15, 3.175), (0.20, 3.836), (0.25, 4.469)):
        assert ratio(m_p) < GAIN_CAP
        assert abs(ratio(m_p) - expect) < 5e-3, (m_p, ratio(m_p))
        assert ratio(0.5) > ratio(m_p)       # 包线会把它们提到 5.0


def test_ratio_monotonic_and_bounded():
    """单调不减、恒在 [1, cap] —— 保证包线上界永远"不小于"真值对应的增益,
    这正是"够大"这一侧的安全性(欠整定会塌带宽,过整定才是要实测的风险)。"""
    xs = np.linspace(0.0, 2.0, 501)
    rs = [ratio(x) for x in xs]
    assert all(b >= a - 1e-12 for a, b in zip(rs, rs[1:]))
    assert rs[0] == 1.0 and max(rs) == GAIN_CAP


if __name__ == '__main__':
    print(f'm_B={p.m}  Jxx={p.Jxx}  arm_d={ARM_D}  cap={GAIN_CAP}')
    print(f'临界载荷 m_p* = {critical_mass():.4f} kg')
    for mp in (0.15, 0.2, 0.25, 0.2937, 0.3, 0.5, 1.0):
        print(f'  m_p={mp:5.3f}  dJ={dJ_from_prior(mp):.5f}  ratio={ratio(mp):.3f}')
    fails = 0
    for n, fn in sorted(globals().items()):
        if n.startswith('test_') and callable(fn):
            try:
                fn(); print(f'  PASS  {n}')
            except AssertionError as e:
                fails += 1; print(f'  FAIL  {n}: {e}')
    raise SystemExit(1 if fails else 0)
