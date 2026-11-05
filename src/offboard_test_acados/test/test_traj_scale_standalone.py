#!/usr/bin/env python3
"""高机动轨迹尺度(D 阶段高机动消融,2026-07-29)的离线验收。

覆盖四件事:
  1) 默认不变性——不传新参数时轨迹逐点等于改造前(r=1.0/w=0.3/ramp=4.0 写死);
  2) 尺度换算——v_peak=√2rw / a_peak=v_peak²/r / yaw_rate峰=YAW_RATE_K·w 与数值扫描吻合;
  3) ramp 冲击判据——auto_ramp_time(w) 下起振段速度不超过稳态峰值;
  4) drop 尖点相位窗口——自适应宽度在高 w 下仍能命中,写死 0.12 会漏。

跟其他 *_standalone.py 一样直接 python3 跑,不需要 ROS 节点/SITL。
"""
import sys
import numpy as np

sys.path.insert(0, '/home/clear/ros2_ws_HJH/src/offboard_test_acados')

from offboard_test_acados.figure8_reference import (
    auto_ramp_time, build_reference_figure8, build_reference_window_figure8)
from offboard_test_acados.acados_params import YAW_RATE_K, p, scaled_stage_terminal_W

fail = []


def check(name, cond, detail=''):
    print(f'{"PASS" if cond else "FAIL"}  {name}  {detail}')
    if not cond:
        fail.append(name)


# --- 1. 默认不变性:不传新参数,轨迹逐点等于改造前的行为 ---
# 改造前 = r=1.0,w=0.3,ramp_time=4.0 写死。这里用显式传参复现"旧行为",
# 与不传参的默认调用逐点比对。
ts = np.arange(0.0, 40.0, 0.05)
old = np.array([build_reference_figure8(t, 1.0, 0.3, 3.0, 0.5, 2.0, 4.0) for t in ts])
new = np.array([build_reference_figure8(t) for t in ts])
check('单点参考默认不变', np.array_equal(old, new),
      f'max|diff|={np.abs(old-new).max():.3e}')

wo = build_reference_window_figure8(5.0, p.N, p.dt)
wn = build_reference_window_figure8(5.0, p.N, p.dt, 1.0, 0.3, 3.0, 0.5, 2.0, 4.0)
check('窗口参考默认不变', np.array_equal(wo, wn))

# 窗口版与单点版在 ramp 段必须一致(新增的 hover/ramp 透传别接错位置参数)
for tq in (1.0, 3.0, 5.5, 12.0):
    w_ = build_reference_window_figure8(tq, p.N, p.dt, r=5.0, w=0.707,
                                        ramp_time=auto_ramp_time(0.707))
    s_ = np.array([build_reference_figure8(tq + i*p.dt, r=5.0, w=0.707,
                                           ramp_time=auto_ramp_time(0.707))
                   for i in range(p.N + 1)]).T
    check(f'窗口/单点一致 t={tq}', np.allclose(w_, s_))

# --- 2. 尺度换算:v_peak / a_peak / 包络,与解析式吻合 ---
print()
print(f'{"r":>5} {"w":>7} {"v_peak":>8} {"a_peak":>8} {"倾角":>7} '
      f'{"yaw_r峰":>8} {"周期":>7} {"包络":>12}')
for r, v_target in [(1.0, None), (0.8, None), (5.0, 2.0), (5.0, 3.0),
                    (5.0, 4.0), (5.0, 5.0), (2.5, 3.0), (10.0, 5.0)]:
    w = (0.3 if r == 1.0 else 0.25) if v_target is None \
        else v_target / (np.sqrt(2) * r)
    ramp = auto_ramp_time(w)
    # 数值扫一整圈稳态段(ramp 之后),取实际峰值
    T = 2 * np.pi / w
    tt = np.linspace(2.0 + ramp, 2.0 + ramp + T, 4000)
    X = np.array([build_reference_figure8(t, r=r, w=w, dz=0.0, ramp_time=ramp)
                  for t in tt])
    v = np.linalg.norm(X[:, 3:6], axis=1)
    yaw_r = np.abs(X[:, 12])
    a_pk = np.sqrt(2) * r * w * w * np.sqrt(2)  # = v_peak^2/r
    tilt = np.degrees(np.arctan2(v.max()**2 / r, 9.81))
    print(f'{r:5.1f} {w:7.3f} {v.max():8.2f} {v.max()**2/r:8.2f} '
          f'{tilt:6.1f}° {yaw_r.max():8.2f} {T:7.1f} '
          f'{2*r:5.1f}x{r:<5.1f}')
    check(f'v_peak 解析式 r={r} w={w:.3f}',
          abs(v.max() - np.sqrt(2)*r*w) < 0.02*np.sqrt(2)*r*w,
          f'数值{v.max():.3f} vs 解析{np.sqrt(2)*r*w:.3f}')
    check(f'yaw_rate峰≈YAW_RATE_Kw r={r} w={w:.3f}',
          abs(yaw_r.max() - YAW_RATE_K*w) < 0.05*YAW_RATE_K*w,
          f'数值{yaw_r.max():.3f} vs 解析{YAW_RATE_K*w:.3f}')

# --- 3. ramp 冲击判据:auto_ramp_time 下起振段速度不超过稳态峰值太多 ---
print()
for r, w in [(5.0, 0.707), (5.0, 0.424), (10.0, 0.354)]:
    for ramp, tag in [(4.0, '写死4.0'), (auto_ramp_time(w), 'auto')]:
        tt = np.linspace(2.0, 2.0 + ramp, 2000)
        X = np.array([build_reference_figure8(t, r=r, w=w, dz=0.0, ramp_time=ramp)
                      for t in tt])
        v_ramp = np.linalg.norm(X[:, 3:6], axis=1).max()
        v_ss = np.sqrt(2) * r * w
        print(f'  r={r} w={w:.3f} ramp={ramp:5.2f}s ({tag:7s}) '
              f'起振峰值={v_ramp:5.2f} 稳态峰值={v_ss:5.2f} '
              f'超出={100*(v_ramp/v_ss-1):+6.1f}%')
    check(f'auto_ramp 起振不超稳态峰值 r={r} w={w:.3f}',
          v_ramp <= 1.05 * v_ss, f'{v_ramp:.3f} vs {v_ss:.3f}')

# --- 4. 权重缩放 ---
print()
Wl, Wel = scaled_stage_terminal_W(1.0, 0.3)
# 与 builder 的构造比对(只有 L_omega 块应不同,见函数 docstring)
Q_ref = p.Q
W_builder = np.block([[Q_ref, np.zeros((Q_ref.shape[0], p.R.shape[1]))],
                      [np.zeros((p.R.shape[0], Q_ref.shape[1])), p.R]])
diff = np.abs(Wl - W_builder)
pos_vel_att_same = np.allclose(Wl[:9, :9], W_builder[:9, :9])
r_same = np.allclose(Wl[12:, 12:], W_builder[12:, 12:])
check('低速下 位置/速度/姿态块 与原权重一致', pos_vel_att_same)
check('R 块不随轨迹尺度变', r_same)
check('低速下 L_omega 块按 docstring 放宽',
      not np.allclose(Wl[9:12, 9:12], W_builder[9:12, 9:12]),
      f'{Wl[9,9]:.2f} vs {W_builder[9,9]:.2f}')
Wh, Weh = scaled_stage_terminal_W(5.0, 0.707)
check('高速 L_vel 放宽', abs(Wh[3, 3] - 1/(5.0*0.707)**2) < 1e-9,
      f'W_vel={Wh[3,3]:.4f} (L_vel={5.0*0.707:.2f})')
check('高速 L_omega 放宽', abs(Wh[9, 9] - 1/(YAW_RATE_K*0.707)**2) < 1e-9,
      f'W_om={Wh[9,9]:.4f} (L_omega={YAW_RATE_K*0.707:.2f})')
check('终端块相对倍数保持 (位置20x/速度1x/姿态2x/角速度0.2x)',
      np.allclose(Weh[:3, :3], 20*Wh[:3, :3]) and
      np.allclose(Weh[3:6, 3:6], Wh[3:6, 3:6]) and
      np.allclose(Weh[6:9, 6:9], 2*Wh[6:9, 6:9]) and
      np.allclose(Weh[9:12, 9:12], 0.2*Wh[9:12, 9:12]))
check('W_stage 形状匹配 builder', Wl.shape == W_builder.shape,
      f'{Wl.shape} vs {W_builder.shape}')
check('W_e 形状 12x12', Weh.shape == (12, 12), f'{Weh.shape}')

# --- 5. drop 尖点窗口:整圈扫描必须恰好命中一次 ---
print()
for w in [0.25, 0.424, 0.566, 0.707, 1.0, 1.5]:
    for win, tag in [(0.12, '写死0.12'), (max(0.12, 3.0*w*p.dt), 'auto')]:
        hits = 0
        tc = 0.0
        # 扫 3 圈,按当前 NMPC 控制帧 p.dt 步进
        while tc < 3 * 2 * np.pi / w:
            a_mod = (w * tc) % (2 * np.pi)
            if 1.5*np.pi <= a_mod < 1.5*np.pi + win:
                hits += 1
            tc += p.dt
        print(f'  w={w:5.3f} win={win:5.3f} ({tag:9s}) 3圈命中={hits}')
    check(f'auto 窗口 3 圈必中 w={w}', hits >= 3, f'hits={hits}')

print()
print(f'{"ALL PASS" if not fail else "FAILED: " + ", ".join(fail)}')
if __name__ == '__main__':
    sys.exit(1 if fail else 0)
