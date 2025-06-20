"""
主程序入口：用于运行主要的物理模拟。
"""
import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numpy.linalg import norm

# 主程序运行：轨道模拟 + 李雅普诺夫指数

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
# 实验示例 1：L1 逃逸轨道
sol_L1 = simulate_orbit('L1', delta=[0, 0, 0, 0.1])
plot_orbit(sol_L1, 'L1')

# 实验示例 2：L2 逃逸轨道
sol_L2 = simulate_orbit('L2', delta=[0, 0, 0, 0.1])
plot_orbit(sol_L2, 'L2')

# 实验示例 3：L4 稳定轨道
sol_L4 = simulate_orbit('L4', delta=[1e-2, 0, 0, 0.0])
C_L4 = compute_jacobi(sol_L4.y[:,0], mu)
plot_orbit(sol_L4, 'L4')

# 实验示例 4：L5 稳定轨道
sol_L5 = simulate_orbit('L5', delta=[1e-2, 0, 0, 0.0])
C_L5 = compute_jacobi(sol_L5.y[:,0], mu)
plot_orbit(sol_L5, 'L5')

# 李雅普诺夫指数估计示例（L4）
state_L4 = [*L_points['L4'], 0.0, 0.0]
t_vals, lyap = estimate_max_lyapunov(mu, state_L4)
plt.plot(t_vals, lyap)
plt.xlabel('时间'); plt.ylabel('局部 Lyapunov 指数'); plt.grid()
plt.title("李雅普诺夫指数估算"); plt.show()
