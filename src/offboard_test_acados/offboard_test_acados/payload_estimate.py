#!/usr/bin/env python3
"""MHE -> NMPC continuous payload-estimate interface.

The ROS transport is ``std_msgs/Float64MultiArray`` so this package does not
need a custom-message build step.  Keeping the schema and the payload algebra
here prevents the estimator and controller from silently disagreeing about
array indices or the meaning of ``s_xy``.
"""

from dataclasses import dataclass

import numpy as np


FIELDS = (
    'm_total', 's_x', 's_y', 'c_x', 'c_y',
    'dJ_xx', 'dJ_yy', 'dJ_zz', 'J_xx', 'J_yy', 'J_zz',
    'no_payload_confidence', 'healthy', 'solution_age_sec',
)
SIZE = len(FIELDS)
LEGACY_SIZE = 12

# 实验计划 §4 的"模型清除逻辑"取值。MHE 与 NMPC 读同名参数 release_baseline,
# 各自实现自己那一半;'P' 是主线,其余取值只在该因素上与 P 不同。
RELEASE_BASELINES = {
    'P': 'Eventless-Full (mainline)',
    'A_PRIME': 'CMD-Immediate: clear on release command, no confirmation',
    'C_SAME': "CMD-Evidence: P's detector, persistence counted only after command",
    'P_NOMOMENT': 'P without the moment-gated release path (mass-domain only)',
    'ORACLE': 'clear at true separation time (upper bound, not deployable)',
}


def _smooth_zero_score(value, full, zero):
    """Return 1 below ``full`` and 0 above ``zero`` with a C1 transition."""
    value = abs(float(value))
    if value <= full:
        return 1.0
    if value >= zero:
        return 0.0
    x = (value - full) / max(zero - full, 1e-12)
    return 1.0 - (3.0 * x * x - 2.0 * x * x * x)


def empty_evidence_scores(m_payload, s_xy, dJ_diag,
                          mass_full=0.015, mass_zero=0.060,
                          moment_full=0.0015, moment_zero=0.0060,
                          inertia_full=0.0020, inertia_zero=0.0100):
    """Return the three independent "empty" scores, unfused.

    Split out so a release *detector* can use the mass and inertia channels
    without inheriting the moment channel's absolute threshold.  That absolute
    threshold is a confidence calibration; a state transition must not depend
    on the estimator's noise floor sitting below it (2026-09-04 deadlock).
    """
    s_norm = float(np.linalg.norm(np.asarray(s_xy, dtype=float)))
    dj_xy = float(np.max(np.abs(np.asarray(dJ_diag, dtype=float)[:2])))
    return (_smooth_zero_score(m_payload, mass_full, mass_zero),
            _smooth_zero_score(s_norm, moment_full, moment_zero),
            _smooth_zero_score(dj_xy, inertia_full, inertia_zero))


NO_PAYLOAD_SCORE_MODES = ('mass', 'legacy_product')


def no_payload_evidence_score(m_payload, s_xy, dJ_diag, mode='mass', **kwargs):
    """Return the release-lifecycle evidence score.

    ``mass`` is the current default. In the deployed 14-state model,
    ``dJ_diag`` is derived deterministically from the same payload-mass value,
    so multiplying a second inertia score cannot add independent evidence.

    ``legacy_product`` preserves the score used by the frozen thesis cohort:
    mass, first-moment and inertia scores are multiplied. Keep this mode for
    exact reproduction of those records or for configurations in which the
    additional quantities come from genuinely independent estimates.
    """
    mode = str(mode).strip().lower()
    if mode not in NO_PAYLOAD_SCORE_MODES:
        raise ValueError(
            f'no-payload score mode {mode!r} not in {NO_PAYLOAD_SCORE_MODES}')
    if mode == 'mass':
        return float(_smooth_zero_score(
            m_payload, kwargs.get('mass_full', 0.015),
            kwargs.get('mass_zero', 0.060)))
    mass_score, moment_score, inertia_score = empty_evidence_scores(
        m_payload, s_xy, dJ_diag, **kwargs)
    return float(np.clip(
        mass_score * moment_score * inertia_score, 0.0, 1.0))


def acc_empty_score(m_acc, m_body, mass_full=0.015, mass_zero=0.060):
    """加速度计力平衡质量 m_acc=T/f_z 的"空载"分(2026-11-01)。

    比力 f_z=(T+D_z)/m 不含重力,所以 m=T/f_z 不经过姿态、速度微分和转动
    模型 —— 与 MHE 的 m̂ 是两条独立的观测路径。载荷质量取正部(比空机还轻
    当空载),阈值与 MHE 质量分同一对。m_acc 不可用(IMU/电机不新鲜、未初始化)
    时返回 0:没有证据就不算空载。
    """
    if m_acc is None or not np.isfinite(m_acc):
        return 0.0
    return _smooth_zero_score(max(float(m_acc) - float(m_body), 0.0),
                              mass_full, mass_zero)


def no_payload_confidence(m_payload, s_xy, dJ_diag, **kwargs):
    """Return the legacy product score used by the frozen thesis cohort.

    The name is retained for log/schema and replay compatibility. This is a
    deterministic evidence score, not a calibrated probability. The inputs
    are not independent in the 14-state mass-only configuration because the
    inertia increment is derived from the same mass estimate.

    New lifecycle code should call :func:`no_payload_evidence_score` and select
    its mode explicitly.
    """
    return no_payload_evidence_score(
        m_payload, s_xy, dJ_diag, mode='legacy_product', **kwargs)


def inertia_from_mass_moment(m_total, s_xy, m_body, J_body, rz,
                             payload_ki=0.0, dJ_xy=None):
    """Return ``(c_xy, dJ_diag, J_diag)`` from ``m`` and ``s=m_P*r_xy``.

    This is the diagonal part of the same first-moment model used by
    :mod:`mhe_model`.  The ill-conditioned O(r_xy**2) terms are deliberately
    omitted there and here.  The off-diagonal terms remain internal to the MHE;
    the current NMPC interface consumes the roll/pitch diagonal increments.
    """
    m_total = max(float(m_total), 1e-6)
    m_payload = max(m_total - float(m_body), 0.0)
    s_xy = np.asarray(s_xy, dtype=float).reshape(2)
    J_body = np.asarray(J_body, dtype=float).reshape(3)
    c_xy = s_xy / m_total
    intrinsic = float(payload_ki) * m_payload
    if dJ_xy is None:
        mu = float(m_body) * m_payload / m_total
        dJ_xy = mu * float(rz) ** 2 + intrinsic
    dJ_diag = np.array([max(float(dJ_xy), 0.0),
                        max(float(dJ_xy), 0.0), intrinsic])
    return c_xy, dJ_diag, J_body + dJ_diag


@dataclass(frozen=True)
class PayloadEstimate:
    m_total: float
    s_xy: np.ndarray
    c_xy: np.ndarray
    dJ_diag: np.ndarray
    J_diag: np.ndarray
    no_payload_confidence: float
    # ``healthy`` is the health of the most recent MHE solve *and* its input
    # streams.  ``solution_age_sec`` is independent of ROS transport age: MHE
    # may keep publishing a held last-good estimate after a failed solve.
    healthy: bool = False
    solution_age_sec: float = 1.0e6

    def as_array(self):
        return np.array([
            self.m_total, *self.s_xy, *self.c_xy, *self.dJ_diag,
            *self.J_diag, self.no_payload_confidence,
            1.0 if self.healthy else 0.0, self.solution_age_sec,
        ], dtype=float)

    @classmethod
    def from_array(cls, values):
        a = np.asarray(values, dtype=float)
        # A rolling deployment may briefly expose a new NMPC to the old
        # 12-field publisher.  Decode it as explicitly unhealthy rather than
        # crashing or, worse, treating an unqualified estimate as fresh.
        if a.shape not in ((SIZE,), (LEGACY_SIZE,)) or not np.all(np.isfinite(a)):
            raise ValueError(
                f'payload estimate must be {SIZE} finite values '
                f'({LEGACY_SIZE}-field legacy frames are accepted unhealthy)')
        healthy = bool(a[12] >= 0.5) if a.shape == (SIZE,) else False
        solution_age = max(float(a[13]), 0.0) if a.shape == (SIZE,) else 1.0e6
        return cls(
            m_total=float(a[0]), s_xy=a[1:3].copy(), c_xy=a[3:5].copy(),
            dJ_diag=a[5:8].copy(), J_diag=a[8:11].copy(),
            no_payload_confidence=float(np.clip(a[11], 0.0, 1.0)),
            healthy=healthy, solution_age_sec=solution_age)


def slew(value, target, max_rate, dt):
    """Scalar/vector slew limiter used by the NMPC model and gain schedule."""
    value = np.asarray(value, dtype=float)
    target = np.asarray(target, dtype=float)
    step = max(float(max_rate), 0.0) * max(float(dt), 0.0)
    out = value + np.clip(target - value, -step, step)
    return float(out) if out.ndim == 0 else out


def headroom_limited_scale(current, command, requested_scale, limit):
    """Limit a common roll/pitch error scale before body-rate saturation.

    The returned scale stays at least one, so scheduling never makes the raw
    NMPC command less authoritative.  A common scale preserves the direction
    of the roll/pitch rate-error vector.
    """
    current = np.asarray(current, dtype=float).reshape(-1)
    command = np.asarray(command, dtype=float).reshape(-1)
    if current.shape != command.shape or not np.all(np.isfinite(current)) \
            or not np.all(np.isfinite(command)):
        raise ValueError('current and command must be equally shaped and finite')
    limit = float(limit)
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError('limit must be positive and finite')
    scale = max(1.0, float(requested_scale))
    for w_now, w_cmd in zip(current, command):
        error = w_cmd - w_now
        if abs(error) <= 1e-12:
            continue
        bound = ((limit - w_now) / error if error > 0.0
                 else (-limit - w_now) / error)
        scale = min(scale, max(1.0, float(bound)))
    return float(scale)


def relative_error_percent(value, reference, eps=1e-6):
    """Finite-safe diagnostic relative error; undefined references yield NaN."""
    reference = float(reference)
    if not np.isfinite(reference) or abs(reference) <= abs(float(eps)):
        return float('nan')
    return float((float(value) - reference) / reference * 100.0)


# --- drop 侧释放判据(2026-09-05 三信息源改造)---------------------------------
# 上一版把 mass 与 inertia 当两票用是**虚假鲁棒性**:连续接口里
# dJ = μ(m_p)·r_z² 完全由 m_est 与固定 r_z 派生,两者携带同一份质量证据。
# 所以这里按**真正独立的三个信息源**组织:
#   ① 一阶矩塌陷   —— 相对 ratio 为主,绝对 |s| 只作 sanity 上限(见下)
#   ② 载荷量证据   —— mass/inertia 合并成**一条**通道(它们不独立)
#   ③ 残差方向证据 —— no-signal detector 的 DROP,独立的动态证据
#
# ⚠️ 绝对 |s| 没有判别力,别拿它当判据:09-04/05 五轮实测,带载段 |s| 最小值
#    0.0039 kg·m 反而**低于**卸载后 |s| 的最大值 0.0049 —— 两个分布是重叠的。
#    它只用来挡"peak 异常大导致 ratio 假性偏小"那种情形,所以阈值给得宽松。

# ★ 0.30 而非 0.20:59 轮 valid 历史日志的留一交叉验证 59/59 折一致选中 0.30
#   (留出轮 90% 成功,0.20 只有 48/59);两个阈值的带载段误释放都是 0——释放要求
#   ratio 与质量通道**同时**成立,带载 ratio 掉到 0.000 的 8 轮全被质量通道否决。
RELEASE_RATIO_THR = 0.30
RELEASE_S_ABS_MAX = 0.008      # kg·m,sanity 上限而非判别器
RELEASE_MASS_MP = 0.03         # kg
RELEASE_SLOW_PERSIST = 5       # 帧,10Hz -> 0.5s
RELEASE_FAST_PERSIST = 2       # 帧
RELEASE_STRONG_PERSIST = 3     # 帧,快速路径 B


def release_evidence(m_payload, s_norm, s_peak, residual_drop,
                     ratio_thr=RELEASE_RATIO_THR,
                     s_abs_max=RELEASE_S_ABS_MAX,
                     mass_release_mp=RELEASE_MASS_MP,
                     residual_strong=False):
    """Return the three independent release indications plus the raw ratio.

    ``residual_drop`` / ``residual_strong`` are the two levels of the release
    residual vote (see the MHE node): the normal vote may only pair with the
    payload-quantity channel, the strong one may pair with a moment collapse.
    """
    s_norm = float(s_norm)
    s_peak = float(s_peak)
    ratio = (s_norm / s_peak) if s_peak > 0.0 else 1.0
    moment_collapsed = bool(ratio < ratio_thr and s_norm < s_abs_max)
    quantity_empty = bool(float(m_payload) < mass_release_mp)
    return {
        'ratio': ratio,
        'moment_collapsed': moment_collapsed,
        'quantity_empty': quantity_empty,      # mass 与 inertia 合并的那一条
        'residual_drop': bool(residual_drop),
        'residual_strong': bool(residual_strong),
    }


def release_decision(ev, load_armed, slow_frames, fast_frames,
                     slow_persist=RELEASE_SLOW_PERSIST,
                     fast_persist=RELEASE_FAST_PERSIST,
                     strong_frames=0,
                     strong_persist=RELEASE_STRONG_PERSIST):
    """Decide whether to release, given evidence and the persistence counters.

    Returns ``(release, why, slow_frames, fast_frames, strong_frames)``.
    Pure function: the caller owns the counters, so the online node and the
    offline replay share exactly one implementation.

    ★ **Every successful release must carry moment evidence.**  Two paths:

      slow  : quantity empty  AND moment collapsed, sustained;
      fastB : strong residual AND moment collapsed.

    2026-09-05 — the third path (plain residual AND quantity empty, no moment)
    was **removed after it caused real premature releases**: 3 of 9 SITL runs
    released while the payload was still attached, the earliest 51.1 s before
    the drop command.  Its design assumption — "quantity_empty is false while
    the payload is on" — is wrong at 4 m/s: ``m_est`` sits on its lower bound
    ``m_min`` for whole stretches of a figure-8, so ``m_p`` is *persistently*
    negative (loaded runs hold ``m_p < 0.03`` for a median of 14 s, worst 48 s).
    Neither amplitude nor persistence separates it from a real drop, and the
    plain residual vote is likewise false-positive during manoeuvres — so the
    pair was two unreliable channels ANDed together, with nothing to veto them.

    The moment channel is the one that measured cleanly: loaded ``ratio``
    bottomed at 0.789 against a 0.30 threshold (2.6x margin) with zero false
    positives across those runs.  So it is now mandatory.

    Cost, accepted deliberately: a drop whose ``ratio`` never collapses is not
    detected here.  That is a *miss*, handled by the unresolved path in the
    controller — never by releasing on weaker evidence.
    """
    if not load_armed:
        return False, '', 0, 0, 0
    strong_ok = ev.get('residual_strong', False) and ev['moment_collapsed']
    slow_ok = ev['quantity_empty'] and ev['moment_collapsed']
    strong_frames = strong_frames + 1 if strong_ok else 0
    slow_frames = slow_frames + 1 if slow_ok else 0
    fast_frames = 0            # 保留字段以免调用方签名变动;fastA 已删除
    if strong_frames >= strong_persist:
        return (True,
                f"fastB: strong residual + moment collapsed "
                f"(ratio={ev['ratio']:.3f}) x{strong_frames}帧",
                slow_frames, fast_frames, strong_frames)
    if slow_frames >= slow_persist:
        return (True,
                f"slow: quantity empty + moment collapsed "
                f"(ratio={ev['ratio']:.3f}) x{slow_frames}帧",
                slow_frames, fast_frames, strong_frames)
    return False, '', slow_frames, fast_frames, strong_frames
