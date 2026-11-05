#!/usr/bin/env python3
"""ω_cmd 缩放(增益调度的 setpoint 侧等效实现)的离线验证(2026-08-25)。

验证的命题:把 setpoint 误差放大 Ĵ_t/J_a,与"把速率环增益乘同样倍数"在**角加速度
层面等价**,从而使闭环特性与真实惯量无关(FlyAware RA-L 2026 的 IAGS 同一原理:
K_k=J_a^{-1}Ĵ_t 使开环传函 independent of J_t)。

⚠️ 今天的教训:离线通过 ≠ SITL 通过(xi 那条线离线五项全过、SITL 当场坠机)。
   这个文件只验证**代数等价性与低通行为**,不声称闭环稳定——那只能靠 SITL。

跑: python3 src/offboard_test_acados/test/test_omega_scale_standalone.py
"""
import sys

import numpy as np

from offboard_test_acados.payload_estimate import headroom_limited_scale

J_A = 0.0142                    # 空机 Jxx
M_B, G = 2.0643, 9.81
ARM = 0.47


def dJ_from_prior(m_p):
    """与 acados_nmpc_node._dJ_from_prior 同一套代数。"""
    m_t = M_B + m_p
    mu = M_B * m_p / m_t if m_t > 0 else 0.0
    return float(mu * ARM ** 2)


def main():
    ok = True

    # --- 1) 代数等价:放大 setpoint 误差 == 放大速率环增益 ---
    print('=== 1) 角加速度层面的等价性 ===')
    print(f'{"m_p":>6} {"J_t":>9} {"s=Ĵt/Ja":>9} {"改增益 ω̇":>11} {"缩setpoint ω̇":>13} {"相对差":>8}')
    K = 0.15                       # 速率环等效 P 增益(任意正值,两边同时出现会约掉)
    w_now, w_cmd = 0.3, 1.0        # 当前角速度 / NMPC 期望角速度
    for m_p in (0.0, 0.15, 0.3, 0.5):
        dJ = dJ_from_prior(m_p)
        J_t = J_A + dJ
        s = J_t / J_A
        # (a) 改增益:τ = (s·K)(ω_cmd − ω),真实角加速度 = τ/J_t
        acc_gain = (s * K) * (w_cmd - w_now) / J_t
        # (b) 缩 setpoint:ω_cmd' = ω + s(ω_cmd − ω),τ = K(ω_cmd' − ω)
        w_eff = w_now + s * (w_cmd - w_now)
        acc_sp = K * (w_eff - w_now) / J_t
        rel = abs(acc_sp - acc_gain) / max(abs(acc_gain), 1e-12) * 100
        good = rel < 1e-6
        ok &= good
        print(f'{m_p:>6.2f} {J_t:>9.5f} {s:>9.3f} {acc_gain:>11.4f} {acc_sp:>13.4f} '
              f'{rel:>7.1e}% {"OK" if good else "**FAIL**"}')
    # 两者都等于**未补偿时空机应有的**角加速度 —— 这才是补偿的目的
    acc_nominal = K * (w_cmd - w_now) / J_A
    print(f'  空机标称 ω̇ = {acc_nominal:.4f} rad/s²;补偿后各载荷下都回到这个值 '
          f'⇒ 闭环特性与 J_t 无关 ✓')

    # --- 2) 未补偿时的退化幅度(说明为什么需要它) ---
    print('\n=== 2) 不补偿时角加速度掉多少 ===')
    for m_p in (0.15, 0.3, 0.5):
        J_t = J_A + dJ_from_prior(m_p)
        loss = (1 - J_A / J_t) * 100
        print(f'  m_p={m_p:.2f}kg → J_t/J_a={J_t/J_A:.2f}×,'
              f'同样 setpoint 误差只得到 {J_A/J_t*100:5.1f}% 的角加速度(掉 {loss:.0f}%)')

    # --- 3) 低通行为:承重渐进时比例连续爬,不阶跃 ---
    print('\n=== 3) 一阶低通(tau=0.5s)跟随承重渐进 ===')
    tau, dt = 0.5, 1.0 / 50.0
    s_now, hist = 1.0, []
    for k in range(int(6.0 / dt)):
        t = k * dt
        # 模拟 attach:t∈[1,3]s 内载荷从 0 线性压到 0.3kg(承重渐进)
        m_p_true = 0.0 if t < 1.0 else min(0.3, 0.3 * (t - 1.0) / 2.0)
        target = np.clip((J_A + dJ_from_prior(m_p_true)) / J_A, 1.0, 5.0)
        s_now += (1 - np.exp(-dt / tau)) * (target - s_now)
        hist.append((t, target, s_now))
    for t_probe in (0.9, 1.5, 2.5, 3.5, 5.9):
        t, tg, sn = min(hist, key=lambda r: abs(r[0] - t_probe))
        print(f'  t={t:4.2f}s  目标 {tg:5.3f}  实际 {sn:5.3f}  '
              f'滞后 {tg - sn:+.3f}')
    final_target, final_s = hist[-1][1], hist[-1][2]
    good = abs(final_s - final_target) < 0.02
    ok &= good
    # 单调性:整个过程不能有反超/振荡(阶跃式改增益的典型病)
    ss = [r[2] for r in hist]
    mono = all(ss[i + 1] >= ss[i] - 1e-9 for i in range(len(ss) - 1))
    ok &= mono
    print(f'  末态收敛到目标: {"OK" if good else "**FAIL**"};'
          f'  全程单调无过冲: {"OK" if mono else "**FAIL**"}')

    # --- 4) headroom guard:调度补偿不能把 raw command 推进硬限幅 ---
    print('\n=== 4) headroom guard(omega_cmd_max=2.0 rad/s) ===')
    WMAX = 2.0
    for m_p in (0.15, 0.3, 0.5):
        s = np.clip((J_A + dJ_from_prior(m_p)) / J_A, 1.0, 5.0)
        w_now = np.array([0.25, 1.50])
        w_raw = np.array([0.80, 0.20])
        s_eff = headroom_limited_scale(w_now, w_raw, s, 0.95 * WMAX)
        w_scaled = w_now + s_eff * (w_raw - w_now)
        good = np.max(np.abs(w_scaled)) <= 0.95 * WMAX + 1e-12
        ok &= good
        print(f'  m_p={m_p:.2f} s_req={s:.2f} s_eff={s_eff:.2f} '
              f'max|ω|={np.max(np.abs(w_scaled)):.3f} '
              f'{"OK" if good else "**FAIL**"}')

    print('\n' + ('全部通过' if ok else '** 有失败项 **'))
    print('注:本文件只验证代数等价与低通行为,**闭环稳定性必须由 SITL 判定**。')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
