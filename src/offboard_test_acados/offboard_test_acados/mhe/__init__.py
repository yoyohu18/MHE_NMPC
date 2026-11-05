"""速度+质量降阶 MHE(本工作空间唯一的估计器)。

模块:
  params          物理常数、窗口、权重、质量边界与健康阈值(含 NMPC 用到的 dt/m_B/m_max/m_min)
  model           acados 模型 x=[v_world, m], p=[q_meas, T_phys]
  solver_builder  AcadosOcp 构建/复用(独立 codegen 目录)
  weights         事件降权时间表(降阶维度)
  estimator       窗口求解核心 + 质量健康监视器(纯 Python,可单测)
  node            ROS2 节点 mhe_node
"""
