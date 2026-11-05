#!/usr/bin/env python3
"""Pure-numpy checks for the continuous MHE -> NMPC payload interface."""

import numpy as np

from offboard_test_acados.payload_estimate import (
    headroom_limited_scale,
    inertia_from_mass_moment,
    no_payload_confidence,
    no_payload_evidence_score,
    PayloadEstimate,
    relative_error_percent,
    slew,
)


def test_round_trip_and_mass_moment_algebra():
    m_b = 2.0643
    m = m_b + 0.2
    s = np.array([0.0, 0.2 * 0.093])
    c, dj, J = inertia_from_mass_moment(
        m, s, m_b, [0.0142, 0.0142, 0.0284], -0.47)
    assert np.allclose(c, s / m)
    assert dj[0] > 0 and dj[1] == dj[0] and dj[2] == 0
    estimate = PayloadEstimate(m, s, c, dj, J, 0.1, True, 0.02)
    decoded = PayloadEstimate.from_array(estimate.as_array())
    assert decoded.m_total == m
    assert np.allclose(decoded.s_xy, s)
    assert np.allclose(decoded.J_diag, J)
    assert decoded.healthy
    assert decoded.solution_age_sec == 0.02


def test_legacy_payload_frame_is_never_assumed_healthy():
    legacy = np.zeros(12)
    legacy[0] = 2.0643
    legacy[11] = 1.0
    decoded = PayloadEstimate.from_array(legacy)
    assert not decoded.healthy
    assert decoded.solution_age_sec > 100.0


def test_first_moment_vetoes_false_empty_mass():
    # Reproduces the important failure mode: mass touches the empty-airframe
    # bound while a clear eccentric-payload moment remains.
    conf = no_payload_confidence(
        0.0, [0.0, 0.018], [0.0, 0.0, 0.0])
    assert conf == 0.0
    assert no_payload_confidence(0.0, [0.0, 0.0], [0.0, 0.0, 0.0]) == 1.0


def test_no_payload_requires_all_three_estimates_to_decay():
    assert no_payload_confidence(
        0.0, [0.0, 0.0], [0.02, 0.02, 0.0]) == 0.0
    mid = no_payload_confidence(
        0.03, [0.0, 0.003], [0.005, 0.005, 0.0])
    assert 0.0 < mid < 1.0


def test_mass_score_ignores_derived_inertia_and_moment():
    """The current score has one evidential input: payload mass."""
    base = no_payload_evidence_score(
        0.03, [0.0, 0.0], [0.0, 0.0, 0.0], mode='mass')
    changed = no_payload_evidence_score(
        0.03, [np.nan, np.nan], [np.nan, np.nan, np.nan], mode='mass')
    assert base == changed
    assert 0.0 < base < 1.0


def test_legacy_score_remains_available_for_frozen_cohort_replay():
    args = (0.03, [0.0, 0.003], [0.005, 0.005, 0.0])
    assert no_payload_evidence_score(*args, mode='legacy_product') == \
        no_payload_confidence(*args)


def test_slew_is_continuous_and_bidirectional():
    x = 0.0
    seq = []
    for _ in range(5):
        x = slew(x, 1.0, max_rate=2.0, dt=0.1)
        seq.append(x)
    assert np.allclose(seq, [0.2, 0.4, 0.6, 0.8, 1.0])
    assert slew(x, 0.0, max_rate=1.0, dt=0.1) == 0.9


def test_omega_scale_uses_headroom_before_hard_saturation():
    now = np.array([0.25, 1.50])
    raw = np.array([0.80, 0.20])
    scale = headroom_limited_scale(now, raw, 5.0, 1.90)
    scaled = now + scale * (raw - now)
    assert 1.0 <= scale < 5.0
    assert np.max(np.abs(scaled)) <= 1.90 + 1e-12
    # One common factor preserves the roll/pitch error direction.
    assert np.allclose(scaled - now, scale * (raw - now))


def test_omega_scale_is_not_reduced_without_need():
    assert headroom_limited_scale([0.0, 0.0], [0.1, -0.2], 5.0, 1.9) == 5.0


def test_zero_thrust_diagnostic_never_divides_by_zero():
    assert np.isnan(relative_error_percent(2.0643, 0.0))
    assert abs(relative_error_percent(2.1, 2.0) - 5.0) < 1e-12


if __name__ == '__main__':
    test_round_trip_and_mass_moment_algebra()
    test_legacy_payload_frame_is_never_assumed_healthy()
    test_first_moment_vetoes_false_empty_mass()
    test_no_payload_requires_all_three_estimates_to_decay()
    test_mass_score_ignores_derived_inertia_and_moment()
    test_legacy_score_remains_available_for_frozen_cohort_replay()
    test_slew_is_continuous_and_bidirectional()
    test_omega_scale_uses_headroom_before_hard_saturation()
    test_omega_scale_is_not_reduced_without_need()
    test_zero_thrust_diagnostic_never_divides_by_zero()
    print('all passed')
