"""
核心数值算法模块：包含各种数值方法实现。
"""
import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numpy.linalg import norm

# 有效势函数 Φ（用于绘制零速度曲线）
def effective_potential(x, y, mu):
    r1 = np.sqrt((x + mu)**2 + y**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2)
    return 0.5*(x**2 + y**2) + mu1/np.sqrt(r1**2) + mu2/np.sqrt(r2**2)

# RTBP 微分方程
def rtbp_rhs(t, state, mu):
    x, y, vx, vy = state
    r1 = np.sqrt((x + mu)**2 + y**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2)
    dUx = x - mu1*(x + mu)/r1**3 - mu2*(x - 1 + mu)/r2**3
    dUy = y - mu1*y/r1**3 - mu2*y/r2**3
    ax = 2*vy + dUx
    ay = -2*vx + dUy
    return [vx, vy, ax, ay]

# 模拟轨道
def simulate_orbit(point_name, delta=[1e-2, 0, 0, 0], t_max=100):
    x0, y0 = L_points[point_name]
    state0 = [x0 + delta[0], y0 + delta[1], delta[2], delta[3]]
    sol = solve_ivp(rtbp_rhs, [0, t_max], state0, args=(mu,),
                    method='DOP853', rtol=1e-12, atol=1e-10)
    return sol
