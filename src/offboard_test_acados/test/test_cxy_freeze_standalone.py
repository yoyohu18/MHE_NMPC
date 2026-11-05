"""cxy_freeze_in_maneuver 的离线单元测试(不起 ROS,用假 self 调方法)。

覆盖:关闭时逐位透传;悬停段锁存均值;机动中保持锁存值;质量门控拒绝锁存;
各条释放路径立即回到在线值。
"""
import types

import numpy as np
import pytest

acn = pytest.importorskip('offboard_test_acados.acados_nmpc_node')
Node = acn.AcadosNMPCNode
M_MIN = acn.mhe_p.m_min


class _Log:
    def __init__(self):
        self.lines = []

    def info(self, s):
        self.lines.append(('I', s))

    def warn(self, s):
        self.lines.append(('W', s))


def _fake(enable=True, window_n=10, frac=0.8):
    s = types.SimpleNamespace()
    s.cxy_freeze_in_maneuver = enable
    s.cxy_freeze_window_n = window_n
    s.cxy_freeze_min_valid_frac = frac
    s.control_mode = 'mhe'
    s.continuous_payload_estimates = True
    s._cxy_hist = []
    s._cxy_frozen = None
    s._payload_target = types.SimpleNamespace(m_total=2.21)
    s._no_payload_latched = False
    s.payload_unresolved = False
    s.grip_drop_pending = False
    s.grip_dropped = False
    s.grip_dynamic_active = False
    log = _Log()
    s.get_logger = lambda: log
    s.log = log
    for name in ('_apply_cxy_freeze', '_cxy_freeze_try_latch',
                 '_cxy_freeze_release_reason'):
        setattr(s, name, types.MethodType(getattr(Node, name), s))
    return s


def _hover(s, c, n=10, fresh=True):
    for _ in range(n):
        out = s._apply_cxy_freeze(np.array(c), fresh)
        assert np.allclose(out, c)


def test_disabled_passthrough():
    s = _fake(enable=False)
    c = np.array([0.003, -0.008])
    assert s._apply_cxy_freeze(c, True) is c
    s._cxy_freeze_try_latch(20.0)
    assert s._cxy_frozen is None


def test_latch_and_hold_during_maneuver():
    s = _fake()
    _hover(s, [0.0, -0.010], n=5)
    _hover(s, [0.0, -0.006], n=5)
    s.grip_dynamic_active = True
    s._cxy_freeze_try_latch(20.0)
    assert np.allclose(s._cxy_frozen, [0.0, -0.008])
    out = s._apply_cxy_freeze(np.array([0.02, 0.02]), True)   # 机动中漂移
    assert np.allclose(out, [0.0, -0.008])


def test_window_only_uses_recent_frames():
    s = _fake(window_n=4)
    _hover(s, [0.05, 0.05], n=10)
    _hover(s, [0.0, -0.004], n=4)
    s.grip_dynamic_active = True
    s._cxy_freeze_try_latch(20.0)
    assert np.allclose(s._cxy_frozen, [0.0, -0.004])


def test_mass_gate_rejects_lower_bound_window():
    s = _fake()
    s._payload_target.m_total = M_MIN          # MHE 贴下界
    _hover(s, [0.0, -0.010])
    s.grip_dynamic_active = True
    s._cxy_freeze_try_latch(20.0)
    assert s._cxy_frozen is None
    assert any(k == 'W' and 'NOT latched' in m for k, m in s.log.lines)
    out = s._apply_cxy_freeze(np.array([0.02, 0.0]), True)
    assert np.allclose(out, [0.02, 0.0])


def test_stale_window_rejected():
    s = _fake()
    _hover(s, [0.0, -0.010], fresh=False)
    s.grip_dynamic_active = True
    s._cxy_freeze_try_latch(20.0)
    assert s._cxy_frozen is None


def test_no_payload_evidence_rejected():
    s = _fake()
    _hover(s, [0.0, -0.010])
    s._no_payload_latched = True
    s.grip_dynamic_active = True
    s._cxy_freeze_try_latch(20.0)
    assert s._cxy_frozen is None


@pytest.mark.parametrize('attr,val', [
    ('grip_drop_pending', True), ('grip_dropped', True),
    ('payload_unresolved', True), ('_no_payload_latched', True),
    ('grip_dynamic_active', False)])
def test_release_paths(attr, val):
    s = _fake()
    _hover(s, [0.0, -0.010])
    s.grip_dynamic_active = True
    s._cxy_freeze_try_latch(20.0)
    assert s._cxy_frozen is not None
    setattr(s, attr, val)
    live = np.array([0.0, -0.001])
    assert np.allclose(s._apply_cxy_freeze(live, True), live)
    assert s._cxy_frozen is None and s._cxy_hist == []
    assert any('RELEASED' in m for _, m in s.log.lines)
