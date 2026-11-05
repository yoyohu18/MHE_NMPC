#!/usr/bin/env python3
# 阶段 A 验证:不需要 ROS2/PX4/Gazebo,只测 acados 求解器本身对不对。
# 运行方式: source install/setup.bash 之后直接 python3 -m pytest 这个文件,
# 或者 python3 test_acados_ocp_standalone.py 直接跑(文件末尾有 __main__ 入口)。
#
# 运行前需要:
#   export ACADOS_SOURCE_DIR=/home/clear/acados
#   export LD_LIBRARY_PATH=/home/clear/acados/lib:$LD_LIBRARY_PATH

import os
import time

import numpy as np

os.environ.setdefault('ACADOS_SOURCE_DIR', '/home/clear/acados')
os.environ['LD_LIBRARY_PATH'] = (
    '/home/clear/acados/lib:' + os.environ.get('LD_LIBRARY_PATH', ''))

from offboard_test_acados.acados_params import p  # noqa: E402
from offboard_test_acados.acados_solver_builder import (  # noqa: E402
    ensure_acados_ocp_solver,
)

HOVER_X = np.array([0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def _set_reference(solver, xr, m=None, dJ=0.0, c=(0.0, 0.0), d=(0.0, 0.0, 0.0)):
    """model.p 现在是 [xr(13); m(1); dJ(1); c_xy(2); d_lumped(3)] 20维(质量/吊挂
    惯量增量/复合质心偏移/平动 lumped 扰动都是运行时参数,见 acados_model.py),
    m 默认用标定常数 p.m、dJ/c/d 默认空机无扰 0,跟改动前的行为一致——既有测试
    本来就是在验证"空机标定质量下"的求解器行为,不是在测自适应参数。
    (2026-11-05:删除转动 lumped 通道 xi,23 维→20 维。)"""
    if m is None:
        m = p.m
    param = np.concatenate([xr, [m], [dJ], c, d])
    for i in range(p.N + 1):
        solver.set(i, 'p', param)


def _solve_to_convergence(solver, max_iters=20):
    """MERIT_BACKTRACKING 线搜索每次只走一部分步长,参考点偏移大的时候一次
    solve() 调用未必能在 nlp_solver_max_iter 以内完全收敛——跟真实 NMPC 循环
    一样,靠多次调用、每次复用上一次的 warm-start 逐步逼近,而不是要求单次
    调用就给出最终解。"""
    status = None
    for _ in range(max_iters):
        status = solver.solve()
        if status == 0:
            return status
    return status


def _seed_initial_guess(solver, x_init):
    """acados 默认的初始迭代值(没设的话是它内部的默认值,不保证是合法的单位
    四元数)对 SQP 不友好,容易卡在 ACADOS_MAXITER 不收敛——原来 CasADi/IPOPT
    版本里 X_init/U_init 也是显式拿悬停状态/悬停推力去铺满整个时域初始化的,这里
    照样做,不能省。"""
    for i in range(p.N):
        solver.set(i, 'x', x_init)
        solver.set(i, 'u', p.u_hover)
    solver.set(p.N, 'x', x_init)


def test_converges_to_hover_thrust():
    """从悬停状态、悬停参考开始解,应该收敛到 u ≈ u_hover,不偏不倚。"""
    solver = ensure_acados_ocp_solver()
    _seed_initial_guess(solver, HOVER_X)
    solver.set(0, 'lbx', HOVER_X)
    solver.set(0, 'ubx', HOVER_X)
    _set_reference(solver, HOVER_X)

    last_u = None
    for _ in range(20):
        status = solver.solve()
        assert status == 0, f'solver failed with status {status}'
        last_u = solver.get(0, 'u')

    assert np.allclose(last_u, p.u_hover, atol=1e-3), (
        f'expected u≈{p.u_hover}, got {last_u}')


def test_thrust_saturates_at_tmax():
    """喂一个远高于当前位置的参考(逼着它拼命爬升),u[0] 必须卡在 Tmax,
    不能超过——这条专门用来抓"忘了设 idxbu 导致约束形同虚设"的情况。"""
    solver = ensure_acados_ocp_solver()
    _seed_initial_guess(solver, HOVER_X)
    solver.set(0, 'lbx', HOVER_X)
    solver.set(0, 'ubx', HOVER_X)

    far_above = HOVER_X.copy()
    far_above[2] += 1000.0  # 参考高度比当前高 1000m,逼近的合理推力需求远超 Tmax
    _set_reference(solver, far_above)

    status = _solve_to_convergence(solver)
    assert status == 0, f'solver failed with status {status}'
    u_opt = solver.get(0, 'u')

    assert u_opt[0] <= p.Tmax + 1e-6, f'thrust {u_opt[0]} exceeds Tmax={p.Tmax}'
    assert u_opt[0] > p.Tmax - 1e-3, (
        f'expected thrust to saturate near Tmax={p.Tmax}, got {u_opt[0]} '
        f'(约束可能没生效,检查 idxbu/lbu/ubu)')


def test_quaternion_norm_drift_within_tolerance():
    """acados 的 ERK 积分器不会像原来手写的 RK4 那样每步重新归一化四元数,
    这里实测一段水平时域上的漂移量,确认在可接受范围内(不是假设,是测出来的)。"""
    solver = ensure_acados_ocp_solver()
    _seed_initial_guess(solver, HOVER_X)
    solver.set(0, 'lbx', HOVER_X)
    solver.set(0, 'ubx', HOVER_X)

    # 参考点稍微偏一点,逼着它产生非零的姿态/角速度指令,而不是停在零输入的退化情况
    moved_ref = HOVER_X.copy()
    moved_ref[0] = 1.0
    _set_reference(solver, moved_ref)

    status = _solve_to_convergence(solver)
    assert status == 0, f'solver failed with status {status}'

    max_drift = 0.0
    for i in range(p.N + 1):
        x_i = solver.get(i, 'x')
        q_norm = np.linalg.norm(x_i[6:10])
        max_drift = max(max_drift, abs(q_norm - 1.0))

    assert max_drift < 1e-3, (
        f'四元数模长漂移 {max_drift:.2e} 超过 1e-3,需要考虑加 con_h_expr 模长约束')


def test_hover_thrust_increases_with_mass_parameter():
    """闭环自适应的核心前提:质量是 model.p 里的运行时参数,不是编译进去的
    常数。注意:OCP 解出的悬停推力不会精确等于 m*g——只有 stage 0 的状态被
    硬约束钉在参考点上,1..N 是自由变量,代价里还有一项基于旧标定质量算出的
    u_hover_const 在拉,真实收敛值是这些项的加权折中,不是简单等式(已经用
    AcadosModel.f_expl_expr 直接验证过 T=m*g 时 vel_dot_z 确实是 0,动力学本身
    没问题,折中是 OCP 代价结构带来的,不是 bug)。这条不去算那个折中点该是
    多少,只验证方向和量级对不对:质量调重之后悬停推力该显著、单调地涨上去,
    涨幅跟 m*g 的理论涨幅同一个量级。"""
    solver = ensure_acados_ocp_solver()

    def hover_thrust_for_mass(m):
        _seed_initial_guess(solver, HOVER_X)
        solver.set(0, 'lbx', HOVER_X)
        solver.set(0, 'ubx', HOVER_X)
        _set_reference(solver, HOVER_X, m=m)
        last_u = None
        for _ in range(20):
            status = solver.solve()
            assert status == 0, f'solver failed with status {status}'
            last_u = solver.get(0, 'u')
        return last_u[0]

    m_heavy = p.m + 1.5  # 模拟抓起约 1.5kg 的包裹
    T_nominal = hover_thrust_for_mass(p.m)
    T_heavy = hover_thrust_for_mass(m_heavy)

    # 标定质量下参考点和 u_hover_const 用的是同一个质量,没有折中可言,
    # 应该精确收敛到 u_hover[0](就是 test_converges_to_hover_thrust 验证的
    # 那个值,这里顺带核对一下没有被这条测试的求解器状态污染)。
    assert np.isclose(T_nominal, p.u_hover[0], atol=1e-2), (
        f'标定质量下悬停推力应精确等于 u_hover[0]={p.u_hover[0]:.3f}N, '
        f'got {T_nominal:.3f}N')

    expected_delta = (m_heavy - p.m) * p.g
    actual_delta = T_heavy - T_nominal
    assert 0.5 * expected_delta < actual_delta < 2.0 * expected_delta, (
        f'质量加重 {m_heavy - p.m:.2f}kg,理论上悬停推力涨幅量级应该是 '
        f'{expected_delta:.2f}N,实际涨了 {actual_delta:.2f}N——质量参数好像'
        f'没真正进到动力学里(或者反过来,涨太多/太少都不对)')


def test_hover_thrust_compensates_lumped_disturbance():
    """C.1 L1 增广的核心前提(2026-07-23,d_lumped 进 model.p 第 18-20 维):
    向下的 lumped 比力 d_z<0(等效"载荷变重但 NMPC 不知道质量变了"——流派 B
    设定)应让悬停推力涨 ≈ m·|d_z|。u_hover 基准已带 d_z(T_hover=m(g−d_z),
    见 acados_model.py),所以这里跟质量测试不同,不存在"旧基准拉扯"的折中,
    可以断言得更紧。顺带验证 d=0 时行为与 17 维版逐位一致(T=u_hover[0])。"""
    solver = ensure_acados_ocp_solver()

    def hover_thrust_for_d(dz):
        _seed_initial_guess(solver, HOVER_X)
        solver.set(0, 'lbx', HOVER_X)
        solver.set(0, 'ubx', HOVER_X)
        _set_reference(solver, HOVER_X, d=(0.0, 0.0, dz))
        last_u = None
        for _ in range(20):
            status = solver.solve()
            assert status == 0, f'solver failed with status {status}'
            last_u = solver.get(0, 'u')
        return last_u[0]

    dz = -1.4      # ≈0.3kg 载荷在 2.064kg 机体上的比力缺口量级
    T_clean = hover_thrust_for_d(0.0)
    T_dist = hover_thrust_for_d(dz)
    assert np.isclose(T_clean, p.u_hover[0], atol=1e-2), (
        f'd=0 时悬停推力应回退到 u_hover[0]={p.u_hover[0]:.3f}N(与 17 维版'
        f'行为一致), got {T_clean:.3f}N')
    expected_delta = p.m * (-dz)
    actual_delta = T_dist - T_clean
    assert 0.8 * expected_delta < actual_delta < 1.2 * expected_delta, (
        f'd_z={dz} 应使悬停推力涨约 m·|d_z|={expected_delta:.3f}N,'
        f'实涨 {actual_delta:.3f}N')


def test_hover_torque_matches_com_offset():
    """质心偏移建模的核心验证:复合质心水平偏移 c=[cx,cy] 时,机体水平定点
    悬停的平衡输入应该是 [m*g, +cy*m*g, -cx*m*g, 0]——推力沿机体 z 轴不过
    质心产生的常值力矩,必须由等大反向的控制力矩抵消(见 acados_model.py
    tau_thrust_com/u_hover_dyn 注释)。cost 的惩罚基准 u_hover_dyn 也带同一项,
    所以平衡点处残差为零,收敛应该是精确的,不是折中。取 0.3kg@偏心0.109m
    实测工况对应的 c≈[0,+1.4]cm。"""
    solver = ensure_acados_ocp_solver()
    m_t = p.m + 0.3
    if p.geom_coupled:
        # 耦合档:几何槛位装 r_p=[rx,ry,rz],c 由模型内部算 c=(m_P/m_T)·r_xy。
        # 要得到同样的 c=[0,0.014] 就反解 ry = c_y·m_T/m_P。
        m_p = m_t - p.m_B
        r_p = (0.0, 0.014 * m_t / m_p, -0.47)
        c = (0.0, (m_p / m_t) * r_p[1])
        _set_reference(solver, HOVER_X, m=m_t, dJ=r_p[0], c=r_p[1:3])
    else:
        c = (0.0, 0.014)
        _set_reference(solver, HOVER_X, m=m_t, dJ=0.05, c=c)
    _seed_initial_guess(solver, HOVER_X)
    solver.set(0, 'lbx', HOVER_X)
    solver.set(0, 'ubx', HOVER_X)

    last_u = None
    for _ in range(20):
        status = solver.solve()
        assert status == 0, f'solver failed with status {status}'
        last_u = solver.get(0, 'u')

    T_hover = m_t * p.g
    expected = np.array([T_hover, c[1] * T_hover, -c[0] * T_hover, 0.0])
    assert np.allclose(last_u, expected, atol=1e-2), (
        f'带质心偏移的悬停平衡输入应为 {expected},got {last_u}——'
        f'推力-质心力矩项可能没进动力学,或 u_hover_dyn 没同步')


def test_solver_cache_hit_is_fast():
    """配置没变的情况下,第二次构建求解器应该走缓存命中,不重新生成/编译 C 代码,
    耗时应该在 1 秒以内(对照第一次跑这个文件时的 10-30 秒编译)。"""
    ensure_acados_ocp_solver()  # 确保第一次该有的产物已经生成过
    t0 = time.time()
    ensure_acados_ocp_solver()
    elapsed = time.time() - t0
    assert elapsed < 1.0, f'cache hit took {elapsed:.2f}s, expected <1s'


if __name__ == '__main__':
    test_converges_to_hover_thrust()
    print('[PASS] test_converges_to_hover_thrust')
    test_thrust_saturates_at_tmax()
    print('[PASS] test_thrust_saturates_at_tmax')
    test_quaternion_norm_drift_within_tolerance()
    print('[PASS] test_quaternion_norm_drift_within_tolerance')
    test_hover_thrust_increases_with_mass_parameter()
    print('[PASS] test_hover_thrust_increases_with_mass_parameter')
    test_hover_thrust_compensates_lumped_disturbance()
    print('[PASS] test_hover_thrust_compensates_lumped_disturbance')
    test_hover_torque_matches_com_offset()
    print('[PASS] test_hover_torque_matches_com_offset')
    test_solver_cache_hit_is_fast()
    print('[PASS] test_solver_cache_hit_is_fast')
    print('\nAll Stage A standalone checks passed.')
