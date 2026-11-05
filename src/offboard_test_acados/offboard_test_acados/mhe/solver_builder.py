#!/usr/bin/env python3
# 速度+质量降阶 MHE 的 AcadosOcp 构建/复用(MHE_MODEL_VARIANT=reduced)。
#
# 与 mhe_solver_builder(full,14 状态)完全独立:模型名 MODEL_NAME、json、codegen
# 目录都带 'reduced_vm',两种变体的缓存互不覆盖(单测 test_mhe_reduced_standalone
# 核对路径不相交)。求解器配置与离线回放 replay_reduced_mhe.py 一致:
# ERK4 / Gauss–Newton / SQP / MERIT_BACKTRACKING / PARTIAL_CONDENSING_HPIPM。

from pathlib import Path

import numpy as np
from scipy.linalg import block_diag
from acados_template import AcadosOcp, AcadosOcpSolver

from .model import MODEL_NAME, build_model
from .params import p as P0


def generated_root() -> Path:
    # 与 NMPC 共用包级 acados_generated_files 目录,模型名区分缓存
    return Path(__file__).resolve().parent.parent / 'acados_generated_files'


def codegen_dir() -> Path:
    return generated_root() / f'{MODEL_NAME}_c_generated_code'


def json_path() -> Path:
    return generated_root() / f'{MODEL_NAME}_acados_ocp.json'


def build_ocp(params=None) -> AcadosOcp:
    P = params or P0
    ocp = AcadosOcp()
    ocp.model = build_model()
    ocp.code_export_directory = str(codegen_dir())

    ocp.solver_options.N_horizon = P.N
    ocp.solver_options.tf = P.N * P.dt

    ocp.cost.cost_type_0 = 'NONLINEAR_LS'
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'
    ocp.cost.W_0 = block_diag(P.R, P.Q, P.Q0)
    ocp.cost.W = block_diag(P.R, P.Q)
    ocp.cost.yref_0 = np.zeros(3 + P.nw + P.nx)
    ocp.cost.yref = np.zeros(3 + P.nw)
    ocp.cost.yref_e = np.zeros(0)
    ocp.cost.Vx_e = np.zeros((0, P.nx))

    ocp.parameter_values = np.array([1.0, 0.0, 0.0, 0.0, P.m_B * P.g])

    ocp.constraints.idxbx = np.array([P.mass_idx])
    ocp.constraints.lbx = np.array([P.m_min_num])
    ocp.constraints.ubx = np.array([P.m_max])

    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 1
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.qp_solver_cond_N = P.N
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
    return ocp


def ensure_solver(force: bool = False, params=None) -> AcadosOcpSolver:
    P = params or P0
    generated_root().mkdir(parents=True, exist_ok=True)
    ocp = build_ocp(P)
    if force:
        solver = AcadosOcpSolver(ocp, json_file=str(json_path()),
                                 generate=True, build=True,
                                 check_reuse_possible=False)
    else:
        solver = AcadosOcpSolver(ocp, json_file=str(json_path()),
                                 generate=False, build=False,
                                 check_reuse_possible=True)
    # 运行期再写一次质量边界:即使复用到一个与 json 不一致的旧 .so(编译前被杀的
    # 已知坑,见记忆 acados-stale-so-reuse),生效边界也以本进程参数为准。
    for j in range(1, P.N):
        solver.constraints_set(j, 'lbx', np.array([P.m_min_num]))
        solver.constraints_set(j, 'ubx', np.array([P.m_max]))
    return solver
