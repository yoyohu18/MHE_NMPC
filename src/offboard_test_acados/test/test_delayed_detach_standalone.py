#!/usr/bin/env python3
"""Regression tests for the command/physical-detachment mismatch injector."""

from unittest.mock import patch

from offboard_test_acados.gripper.proximity_gripper_node import (
    ProximityGripperNode,
)


class _Logger:
    def warn(self, _msg):
        pass


class _Stub:
    schedule_detach = ProximityGripperNode.schedule_detach
    flush_pending_detach = ProximityGripperNode.flush_pending_detach

    def __init__(self, delay):
        self.detach_delay_sec = delay
        # jam 注入(2026-09-15)与 detach_delay_sec>0 在产品里互斥(见节点 135 行
        # 的 raise),本文件测的是 delay 路径,只能是 False。
        self.detach_jam = False
        self.attached = {'box'}
        self._pending_detach = {}
        self.sent = []

    def get_logger(self):
        return _Logger()

    def send_detach(self, tgt):
        self._pending_detach.pop(tgt, None)
        self.attached.discard(tgt)
        self.sent.append(tgt)


def test_zero_delay_preserves_normal_behavior():
    stub = _Stub(0.0)
    stub.schedule_detach('box', 'enable=false')
    assert stub.sent == ['box']


def test_nonzero_delay_holds_then_detaches_once():
    stub = _Stub(4.0)
    with patch('offboard_test_acados.gripper.proximity_gripper_node.time.monotonic',
               return_value=10.0):
        stub.schedule_detach('box', 'enable=false')
        stub.schedule_detach('box', 'repeat')
    assert stub.sent == []
    assert stub._pending_detach == {'box': 14.0}
    with patch('offboard_test_acados.gripper.proximity_gripper_node.time.monotonic',
               return_value=13.99):
        stub.flush_pending_detach()
    assert stub.sent == []
    with patch('offboard_test_acados.gripper.proximity_gripper_node.time.monotonic',
               return_value=14.0):
        stub.flush_pending_detach()
        stub.flush_pending_detach()
    assert stub.sent == ['box']


if __name__ == '__main__':
    test_zero_delay_preserves_normal_behavior()
    test_nonzero_delay_holds_then_detaches_once()
    print('[PASS] delayed-detach fault injector')
