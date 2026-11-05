#!/usr/bin/env python3
# 搭 AcadosOcp 并构建/复用求解器。
#
# acados_template 自带一套代码复用检测(AcadosOcpSolver.is_code_reuse_possible):
# 它会把整个 OCP 配置算一个哈希存进生成的 json 里,下次构建时比对,配置没变就直接跳过
# C 代码生成+编译(只在改了 Q/R/P/N/dt/约束等任何一项时才会真的重新生成,耗时 10-30 秒)。
# 这比我们自己再搭一套源码哈希/stamp 文件更准——它哈希的是真正生效的 OCP 对象,不是
# 文件字节,不会有"改了文件但 OCP 其实没变"或者反过来的误判。我们只需要把
# code_export_directory 固定到包内一个路径(默认是相对 cwd 的 c_generated_code/,
# 每次从不同目录 ros2 run 都会重新生成,必须钉死)。

from pathlib import Path

import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver

from .acados_model import MODEL_NAME, build_acados_model
from .acados_params import p


def generated_root() -> Path:
    return Path(__file__).resolve().parent / 'acados_generated_files'


def codegen_dir() -> Path:
    return generated_root() / f'{MODEL_NAME}_c_generated_code'


def json_path() -> Path:
    return generated_root() / f'{MODEL_NAME}_acados_ocp.json'


def build_ocp() -> AcadosOcp:
    model = build_acados_model()

    ocp = AcadosOcp()
    ocp.model = model
    ocp.code_export_directory = str(codegen_dir())

    nx = p.nx
    ny = model.cost_y_expr.size()[0]      # 16
    ny_e = model.cost_y_expr_e.size()[0]  # 12

    ocp.solver_options.N_horizon = p.N
    ocp.solver_options.tf = p.N * p.dt

    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    ocp.cost.W = np.block([
        [p.Q, np.zeros((p.Q.shape[0], p.R.shape[1]))],
        [np.zeros((p.R.shape[0], p.Q.shape[1])), p.R],
    ])
    ocp.cost.W_e = p.P
    # yref 全是零,因为 cost_y_expr/cost_y_expr_e 本身就已经是误差了(tracking_error_sym
    # 算出来的就是 x 减参考的结果),不是绝对状态——跟 yref 相减就是"误差再减零误差"。
    ocp.cost.yref = np.zeros(ny)
    ocp.cost.yref_e = np.zeros(ny_e)

    # 必须显式设置,否则 acados 会在 OCP 构建时报维度不一致的错。注意质量维
    # 不是零——model.p 现在是 [xr(13); m(1); dJ(1); c_xy(2); d(3)] 20维,质量这一维
    # 如果留 0 会在动力学里被 1/m 除,直接除零崩掉;dJ(吊挂惯量增量)和 c_xy
    # (复合质心水平偏移)默认 0=空机。这里只是构建时的占位默认值,实际运行时
    # acados_nmpc_node 每次 solve 前都会用当前估计覆盖(见 solve_nmpc)。
    # [xr(13)=0; m=标定值; dJ=0; c_xy=0,0; d_lumped(3)=0] = 20维
    # (d 见 acados_model.py;lumped 通道默认全零 → 关掉时逐位退回历史行为)
    ocp.parameter_values = np.concatenate(
        [np.zeros(nx), [p.m], [0.0, 0.0, 0.0], np.zeros(3)])

    # 输入边界:必须连 idxbu 一起设,漏了的话 acados 不报错,但约束形同虚设
    ocp.constraints.lbu = np.array([p.Tmin, -p.tau_max, -p.tau_max, -p.tau_psi])
    ocp.constraints.ubu = np.array([p.Tmax, p.tau_max, p.tau_max, p.tau_psi])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    # 初始状态约束占位,每次 solve 前都会用 lbx/ubx 在 stage 0 覆盖成当前状态
    ocp.constraints.x0 = np.array(
        [0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 1
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.qp_solver_cond_N = p.N
    # 换回完整 SQP(RTI 不报错但会悄悄发散,见 acados_nmpc_node 实测记录)。
    # 使用完整 SQP；当前 dt=0.05/N=20，预测时域仍为 1.0s。
    ocp.solver_options.nlp_solver_type = 'SQP'
    # 默认 globalization='FIXED_STEP'(每次都走满步长,没有线搜索)在四元数这种
    # 非线性代价项上实测会卡进一个两点来回振荡的死循环(alpha 一直是 1.0,
    # res_stat 在两个值之间反复跳,永远不收敛)。换成带回溯线搜索的
    # MERIT_BACKTRACKING 解决了这个问题。
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING'

    return ocp


def ensure_acados_ocp_solver(force: bool = False) -> AcadosOcpSolver:
    generated_root().mkdir(parents=True, exist_ok=True)
    ocp = build_ocp()
    if force:
        return AcadosOcpSolver(ocp, json_file=str(json_path()),
                                generate=True, build=True, check_reuse_possible=False)
    # generate=False, build=False 才会真的触发 is_code_reuse_possible 检查;
    # 如果 ocp 配置(N/dt/Q/R/P/约束/模型...)跟上次生成时一致就跳过重新生成+编译,
    # 否则 acados 会在内部自动把 generate/build 改回 True 并重新生成——所以这一个
    # 调用同时覆盖"缓存命中"和"配置变了需要重建"两种情况。
    return AcadosOcpSolver(ocp, json_file=str(json_path()),
                            generate=False, build=False, check_reuse_possible=True)
