#!/usr/bin/env python3
# 执行器推力包线一致性,2026-09-10。
#
# 背景:NMPC 的推力箱式约束原本是"按飞机重量拍"的 Tmin=0.5N / Tmax=2·m_B·g=40.50N,
# 跟执行器实际能发出多少推力无关。而 publish_attitude 把归一化油门夹在 [0.05,0.95],
# 对应真实推力只有 [1.27, 31.35]N。于是 NMPC 以为自己有 40.50N 可用、实际最多
# 31.35N —— 带 0.3kg 载荷时它以为余量 17.31N(0.75g),真实只有 8.16N(0.35g),
# **规划的加速度能力是实际的 2.1 倍**。规划跟不上 → 误差扩大 → 下一拍要更多推力
# → 仍被 clip,正是"闭环失稳而非求解器发散"所需的正反馈条件。
#
# 这个文件锁住的不是某个数值,是**两处必须同源**这件事:约束边界(acados 的
# lbu/ubu)与油门 clip 边界(publish_attitude)。将来谁改了 SIM_GZ_EC_MIN/MAX、
# motorConstant 或那个 0.95,漏改另一处就会在这里失败。
#
# 跑法:
#   export ACADOS_SOURCE_DIR=/home/clear/acados
#   export LD_LIBRARY_PATH=/home/clear/acados/lib:$LD_LIBRARY_PATH
#   python3 test/test_actuator_envelope_standalone.py

import os
import sys

from offboard_test_acados.acados_params import (
    OMEGA_MIN, OMEGA_SPAN, THROTTLE_MAX, THROTTLE_MIN, THRUST_BOX_LEGACY,
    THRUST_K, T_ACT_MAX, T_ACT_MIN, p, thrust_at_throttle,
)


def test_box_matches_actuator_envelope():
    """箱式约束 == 执行器在 clip 边界上能发出的推力。"""
    if THRUST_BOX_LEGACY:
        print('[SKIP] NMPC_THRUST_BOX_LEGACY=1,箱式约束是历史值,不做一致性检查')
        return
    assert abs(p.Tmax - thrust_at_throttle(THROTTLE_MAX)) < 1e-9, (
        f'Tmax={p.Tmax} 与 95% 油门推力 {thrust_at_throttle(THROTTLE_MAX)} 不一致')
    assert abs(p.Tmin - thrust_at_throttle(THROTTLE_MIN)) < 1e-9, (
        f'Tmin={p.Tmin} 与 5% 油门推力 {thrust_at_throttle(THROTTLE_MIN)} 不一致')
    print(f'[PASS] test_box_matches_actuator_envelope '
          f'({p.Tmin:.3f} ~ {p.Tmax:.3f} N)')


def test_box_within_hardware_limit():
    """约束上限不得超过硬件极限(100% 油门)。

    legacy 档反过来断言它**确实**越界 —— 那正是这个档要复现的历史缺陷,
    在这里如实记一笔比让测试变红有用。
    """
    hw_max = thrust_at_throttle(1.0)
    if THRUST_BOX_LEGACY:
        assert p.Tmax > hw_max, 'legacy 档的 Tmax 居然没越界?包线常数被改过了'
        print(f'[PASS] test_box_within_hardware_limit '
              f'(legacy 已知缺陷:Tmax {p.Tmax:.3f} > 硬件 {hw_max:.3f} N,'
              f'超 {100 * (p.Tmax / hw_max - 1):.1f}%)')
        return
    assert p.Tmax <= hw_max + 1e-9, (
        f'Tmax={p.Tmax:.3f}N 超过硬件极限 {hw_max:.3f}N —— 优化器会规划出'
        f'执行器根本发不出的推力')
    assert p.Tmin >= thrust_at_throttle(0.0) - 1e-9
    print(f'[PASS] test_box_within_hardware_limit '
          f'(Tmax {p.Tmax:.3f} <= 硬件 {hw_max:.3f} N)')


def test_node_clip_uses_same_source():
    """publish_attitude 的 clip 边界与 acados_params 必须是同一个对象。

    只做静态检查(import 节点要拖进 rclpy/acados,这里不值得)。读源码确认
    那三个常数是引用而不是各存一份字面量。
    """
    src = os.path.join(os.path.dirname(__file__), '..',
                       'offboard_test_acados', 'acados_nmpc_node.py')
    with open(src, encoding='utf-8') as fh:
        text = fh.read()
    for name in ('ACT_THROTTLE_MIN', 'ACT_THROTTLE_MAX', 'ACT_THRUST_K',
                 'ACT_OMEGA_MIN', 'ACT_OMEGA_SPAN'):
        assert name in text, f'{name} 没有从 acados_params 引入'
    assert 'np.clip(norm, 0.05, 0.95)' not in text, (
        'publish_attitude 又把 clip 边界写成字面量了 —— 必须用 '
        'ACT_THROTTLE_MIN/MAX,否则约束与执行会再次脱钩')
    print('[PASS] test_node_clip_uses_same_source')


def test_hover_margin_is_honest():
    """带载悬停余量必须按真实包线算,不能再是 2mg 那个虚数。"""
    if THRUST_BOX_LEGACY:
        print('[SKIP] legacy 档')
        return
    m_t = p.m_B + 0.30                      # 包线上限载荷
    hover = m_t * p.g
    assert p.Tmax > hover, (
        f'满包线载荷 {m_t:.3f}kg 悬停需 {hover:.2f}N,超过执行器上限 '
        f'{p.Tmax:.2f}N —— 飞机根本悬不住,包线定义有问题')
    margin_g = (p.Tmax - hover) / hover
    assert margin_g < 0.5, (
        f'真实余量 {margin_g:.2f}g 看起来过大,包线常数可能被改错了')
    print(f'[PASS] test_hover_margin_is_honest '
          f'(满载 {m_t:.3f}kg 真实余量 {p.Tmax - hover:.2f}N = {margin_g:.2f}g)')


if __name__ == '__main__':
    print(f'=== legacy={THRUST_BOX_LEGACY} THRUST_K={THRUST_K:.6e} '
          f'omega=[{OMEGA_MIN:.0f},{OMEGA_MIN + OMEGA_SPAN:.0f}] '
          f'throttle=[{THROTTLE_MIN},{THROTTLE_MAX}] '
          f'T_ACT=[{T_ACT_MIN:.3f},{T_ACT_MAX:.3f}]N ===')
    test_box_matches_actuator_envelope()
    test_box_within_hardware_limit()
    test_node_clip_uses_same_source()
    test_hover_margin_is_honest()
    print('\nall passed')
    sys.exit(0)
