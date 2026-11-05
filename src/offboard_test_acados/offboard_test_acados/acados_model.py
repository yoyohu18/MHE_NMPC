#!/usr/bin/env python3
# acados 专用动力学。质量 m 不是常数 p.m，而是 model.p 里的运行时参数——
# 这样 mhe_node 估出来的质量才能真的喂给 NMPC 用,不是只发个诊断话题。
#
# model.p = [xr(13维,参考状态); m(1维,当前质量估计); dJ(1维,吊挂惯量增量);
#            c_xy(2维,复合质心在机体系的水平偏移); d_lumped(3维,平动 lumped
#            扰动比力,C.1 L1 增广用)] = 20维。(2026-11-05 删除了 08-25 试验过、
#            判负且从未在任何报告批次中启用的转动 lumped 通道 xi_lumped(3维),
#            原为 23维。)时变参考用 model.p
# 而不是 yref 来传,是因为代价里的四元数误差项对参考值是非线性的(四元数乘法),
# 没法写成简单的 "y - yref" 形式,必须让参考值进 CasADi 表达式本身——质量/
# 惯量/质心参数顺路放在同一个 model.p 向量里。

import casadi as cs
from acados_template import AcadosModel

from .common import tracking_error_sym

from .acados_params import p

# 耦合档必须换 codegen 目录名(见 acados_solver_builder),否则两档共用生成代码。
# 姿态误差 'euclid' 档同理换名(维数不同,不能共用生成代码)。
MODEL_NAME = (('offboard_test_acados_nmpc_coupled' if p.geom_coupled
               else 'offboard_test_acados_nmpc')
              + ('_qeuc' if p.quat_err == 'euclid' else ''))


def build_acados_model():
    x_sym = cs.MX.sym('x', p.nx)
    u_sym = cs.MX.sym('u', p.nu)
    xr_sym = cs.MX.sym('xr', p.nx)
    m_sym = cs.MX.sym('m', 1)
    # 吊挂载荷对滚转/俯仰惯量的增量 dJ = m_p*d^2(点质量 m_p 刚性焊在机体正
    # 下方 d 处,平行轴定理;载荷在机体 z 轴上,Izz 不变,CoM 沿轴下移也不产生
    # 推力/重力力矩——推力线始终过合成 CoM)。空机时 0。这是夹爪实验里"0.2kg
    # 就把姿态打崩"的主导缺失物理:0.3kg@0.47m 臂 → dJ≈0.066,是空机
    # Jxx(0.0217) 的 4 倍,NMPC 按空机惯量规划的角加速度真机做不到,力矩饱和
    # 后 SQP 级联发散。跟 m_sym 一样走 model.p,attach 时随质量阶跃一起喂。
    dJ_sym = cs.MX.sym('dJ', 1)
    # 复合质心(机体+载荷)在机体系的水平偏移 c=[cx,cy] [m]。0.3kg@0.47m 的 dJ
    # 建模实验(2026-07-02 20:21)证明仅补惯量不够:attach 实测 d_xy=0.109m 的
    # 水平偏心让复合质心侧移约 1.4cm,悬停推力 ~23N 沿机体 z 轴穿过机体原点、
    # 不再过复合质心,产生 ~0.32Nm 的恒定 roll 力矩——tau_max=0.5 的 64%,日志
    # 里 t=1.81s(box 刚离地)pos_err 才 0.022m roll 就 100% 饱和,随后级联发散。
    # 把这个力矩显式建进 om_dot,NMPC 才知道"带偏心载荷悬停本来就要一个常值
    # 力矩",而不是把它当成需要用姿态机动去消灭的异常。空机时 0。
    c_sym = cs.MX.sym('c_xy', 2)
    # 平动 lumped 扰动比力 d [m/s²](C.1 L1-NMPC baseline,2026-07-23,实施计划
    # §3/D1 拍板"进模型"):L1 增广在线估计的未建模平动效应(流派 B——变载荷
    # 不显式估质量,当 lumped 扰动补偿;Hanover RA-L 2021 同款位置)。直接加在
    # vel_dot 上(比力量纲,与质量假设解耦——换算成力要乘质量,而"质量是多少"
    # 正是两个流派的分歧点,见 l1_adaptive.py 量纲约定)。主方法(truth/online
    # 模式)恒零,行为与 17 维版逐位一致。
    d_sym = cs.MX.sym('d_lumped', 3)

    vel = x_sym[3:6]
    q_  = x_sym[6:10]
    om  = x_sym[10:13]
    T_   = u_sym[0]
    tau_ = u_sym[1:4]

    qw = q_[0]; qx = q_[1]; qy = q_[2]; qz = q_[3]
    R_q = cs.vertcat(
        cs.horzcat(qw**2+qx**2-qy**2-qz**2, 2*(qx*qy-qw*qz),        2*(qx*qz+qw*qy)),
        cs.horzcat(2*(qx*qy+qw*qz),          qw**2-qx**2+qy**2-qz**2, 2*(qy*qz-qw*qx)),
        cs.horzcat(2*(qx*qz-qw*qy),          2*(qy*qz+qw*qx),         qw**2-qx**2-qy**2+qz**2)
    )

    g_vec = cs.MX([0.0, 0.0, p.g])
    # 跟 build_dynamics() 唯一的数学差异:用 model.p 里的 m_sym(变量)而不是
    # 常数 p.m——这一行就是闭环自适应的关键,质量估计变了,这里的加速度
    # 映射立刻跟着变,NMPC 下一次求解就会用新的质量重新规划推力。
    vel_dot = (1.0/m_sym) * (cs.mtimes(R_q, cs.vertcat(0.0, 0.0, T_)) - p.kd * vel) \
        - g_vec + d_sym

    Xi_q = cs.vertcat(
        cs.horzcat(-qx, -qy, -qz),
        cs.horzcat( qw, -qz,  qy),
        cs.horzcat( qz,  qw, -qx),
        cs.horzcat(-qy,  qx,  qw)
    )
    quat_dot = 0.5 * cs.mtimes(Xi_q, om)

    if p.geom_coupled:
        # 耦合档(2026-08-24):dJ_sym/c_sym 两个槛位改装载荷几何偏移 r_p=[rx,ry,rz]
        # (第 15..17 维,维数不变),J 与 c 由质量槛位 m_sym 现算——与
        # mhe_model.py 耦合档同一套代数,两边必须一致,否则 NMPC 的模型与 MHE 的
        # 模型对同一个质量给出不同的姿态动力学。
        r_p = cs.vertcat(dJ_sym, c_sym)          # [rx, ry, rz]
        dm = m_sym - p.m_B
        m_p = 0.5 * (dm + cs.sqrt(dm * dm + p.mp_pos_eps ** 2))
        mu = p.m_B * m_p / m_sym
        J_mat = cs.diag(cs.vertcat(p.Jxx, p.Jyy, p.Jzz)) \
            + p.payload_ki * m_p * cs.MX.eye(3) \
            + mu * (cs.dot(r_p, r_p) * cs.MX.eye(3) - cs.mtimes(r_p, r_p.T))
        c_eff = (m_p / m_sym) * r_p[0:2]
        Jom = cs.mtimes(J_mat, om)
        tau_thrust_com = cs.vertcat(-c_eff[1] * T_, c_eff[0] * T_, 0.0)
        om_dot = cs.mtimes(cs.inv(J_mat),
                           tau_ + tau_thrust_com - cs.cross(om, Jom))
    else:
        # legacy(默认):表达式逐字保持原样,生成的 C 代码与历史批次一致。
        c_eff = c_sym
        J_vec = cs.vertcat(p.Jxx + dJ_sym, p.Jyy + dJ_sym, p.Jzz)
        Jom   = J_vec * om
        # 推力作用在机体原点(桨盘中心)沿机体 z 轴,对复合质心(位于 r_c=[cx,cy,cz])
        # 的力矩 = (-r_c)×[0,0,T] = [-cy*T, +cx*T, 0]。重力作用点就是质心,对质心
        # 无力矩;cz(质心竖向下移)不跟沿 z 的推力叉出力矩,所以只需要 cx,cy 两维。
        tau_thrust_com = cs.vertcat(-c_sym[1] * T_, c_sym[0] * T_, 0.0)
        om_dot = (tau_ + tau_thrust_com - cs.cross(om, Jom)) / J_vec

    xdot = cs.vertcat(vel, vel_dot, quat_dot, om_dot)

    model = AcadosModel()
    model.name = MODEL_NAME
    model.x = x_sym
    model.u = u_sym
    model.p = cs.vertcat(xr_sym, m_sym, dJ_sym, c_sym, d_sym)
    model.f_expl_expr = xdot

    # 12维 [ep;ev;eq_vec;eomega]('vec')或 13维 [ep;ev;q−q_ref;eomega]('euclid')
    e_track = tracking_error_sym(x_sym, xr_sym, p.quat_err)
    # u_hover 用 m_sym(已经是 model.p 的一部分)实时算,不再是基于固定标定
    # 质量 p.m 的编译时常量。早先(质量阶跃只有"抓取"场景、且单元测试用方向
    # 性检查代替精确相等断言时)判断过"惩罚基准用旧值不会让控制失效,影响
    # 很小,暂不处理"——这个判断在"投放"场景里被推翻了:满载 3.5kg 时真实
    # 悬停推力约 34N,但写死的基准还停在空载标定值约 20N,cost 里 u-u_hover
    # 这一项持续把推力"拽"向这个偏低的旧基准,实测会让追踪误差卡在 0.85m+
    # 完全不收敛,不是"影响很小"。改成 m_sym*g 之后,质量估计变了,惩罚中心
    # 跟着动态平移,不再固化一个过时的悬停推力假设。
    # 带偏心载荷时,定点悬停要求机体水平(推力必须竖直才能不平移),此时推力
    # 对复合质心的力矩不为零,必须由常值力矩 tau=[+cy*T, -cx*T, 0] 抵消(即
    # om_dot 里 tau_thrust_com 的相反数)。惩罚基准不带这一项的话,R 会持续把
    # 力矩往 0 拽,跟"u_hover 用过时质量基准把推力拽低"是同一类稳态偏差。
    # d 进模型后悬停推力平衡变为 T/m − g + d_z = 0 → T_hover = m·(g − d_z)。
    # 惩罚基准不带 d_z 的话,L1 模式下(如 attach 后 d_z<0)R 会持续把推力拽回
    # 无扰动基准 m·g——跟上面"u_hover 用过时质量把推力拽低"是同一类稳态偏差,
    # 这里在引入 d 的同时一并处理,不等实测踩坑。水平分量 d_x/d_y 由姿态倾斜
    # 抵消,不进推力/力矩基准。truth/online 模式 d≡0,该式退化回 m·g 不变。
    T_hover = m_sym * (p.g - d_sym[2])
    u_hover_dyn = cs.vertcat(
        T_hover, c_eff[1] * T_hover, -c_eff[0] * T_hover, 0.0)
    model.cost_y_expr = cs.vertcat(e_track, u_sym - u_hover_dyn)     # 16维('euclid' 17)
    model.cost_y_expr_e = e_track                                     # 12维('euclid' 13)

    return model
