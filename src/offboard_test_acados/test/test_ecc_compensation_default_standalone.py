#!/usr/bin/env python3
"""NMPC 侧偏心补偿默认关(2026-10-20)的契约检查。

09-24 消融(paper/data/ecc_ablation_20260924_202511)显示 NMPC 的 c_xy 对跟踪无可测
收益,用户 10-20 拍板默认去掉。必须同时成立的三件事:
  1. 默认下 solver 几何槽的 cx/cy 恒为 0,dJ 照常传(内环/惯量补偿不能跟着丢);
  2. 显式 eccentricity_compensation=True 时逐位回到旧行为(复现冻结批次用);
  3. 生命周期(释放确认)不读 NMPC 的 c_est —— MHE 的一阶矩 s 仍进置信度。
"""

import inspect

import numpy as np

from offboard_test_acados import acados_nmpc_node as an
from offboard_test_acados.payload_estimate import no_payload_confidence


class _Stub:
    def __init__(self, **kw):
        self.continuous_payload_estimates = True
        self.control_mode = 'mhe'
        self.dJ_est = 0.0421
        self.c_est = np.array([0.0031, -0.0075])
        self.__dict__.update(kw)


def _slot(stub):
    return an.AcadosNMPCNode._geom_slot(stub)


def test_declared_default_is_off():
    src = inspect.getsource(an.AcadosNMPCNode.__init__)
    assert "'eccentricity_compensation', False" in src


def test_default_zeroes_c_keeps_dj():
    # 没设属性的桩走 getattr 兜底,兜底值必须与产品默认一致(关)
    out = _slot(_Stub())
    np.testing.assert_array_equal(out, [0.0421, 0.0, 0.0])


def test_explicit_on_restores_legacy_slot():
    out = _slot(_Stub(eccentricity_compensation=True))
    np.testing.assert_array_equal(out, [0.0421, 0.0031, -0.0075])


def test_explicit_off_zeroes_c():
    out = _slot(_Stub(eccentricity_compensation=False))
    np.testing.assert_array_equal(out, [0.0421, 0.0, 0.0])


def test_release_confidence_still_uses_moment():
    # 带载偏心的一阶矩必须仍能否决"空载"——这是生命周期安全靠的那条证据,
    # 与 NMPC 是否把 c_xy 放进模型无关。
    m_p = 0.02                     # 质量域已接近空载
    dJ = np.zeros(3)
    empty = no_payload_confidence(m_p, np.zeros(2), dJ)
    loaded_moment = no_payload_confidence(m_p, np.array([0.0, -0.016]), dJ)
    assert loaded_moment < 0.5 * empty


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
