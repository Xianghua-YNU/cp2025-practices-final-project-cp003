"""
数据后处理与分析模块：用于对模拟结果进行分析。
"""
import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numpy.linalg import norm

# 模块一：拉格朗日点求解（Levenberg-Marquardt 方法）
# 系统参数（地月系统）
mu = 0.0123
eps = 1e-10  # 容差

# 旋转势函数梯度
def grad_phi(pos, mu):
    x, y = pos
    r1 = np.sqrt((x + mu)**2 + y**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2)
    dUx = x - (1 - mu)*(x + mu)/r1**3 - mu*(x - 1 + mu)/r2**3
    dUy = y - (1 - mu)*y/r1**3 - mu*y/r2**3
    return np.array([dUx, dUy])

# 拉格朗日点求解（给定初值）
initial_guesses = {
    'L1': [0.5, 0.0], 'L2': [1.1, 0.0], 'L3': [-1.1, 0.0],
    'L4': [0.48785, +np.sqrt(3)/2], 'L5': [0.48785, -np.sqrt(3)/2]
}
L_points = {}
for name, guess in initial_guesses.items():
    sol = root(grad_phi, guess, args=(mu,), method='lm', tol=eps)
    L_points[name] = sol.x
    print(f"{name} = {sol.x}")

# 设置系统参数（地月系统）
mu = 0.0123
eps = 1e-10  # 容差

# 理论解析解（近似）
L_theory = {
    'L1': [0.836915, 0.0],
    'L2': [1.15568, 0.0],
    'L3': [-1.00506, 0.0],
    'L4': [0.5 - mu, +np.sqrt(3)/2],
    'L5': [0.5 - mu, -np.sqrt(3)/2],
}

# 旋转势函数的梯度 ∇Φ*
def grad_phi(pos, mu):
    x, y = pos
    r1 = np.sqrt((x + mu)**2 + y**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2)
    dUx = x - (1 - mu)*(x + mu)/r1**3 - mu*(x - 1 + mu)/r2**3
    dUy = y - (1 - mu)*y/r1**3 - mu*y/r2**3
    return np.array([dUx, dUy])

# 初始猜测（来自文献或几何近似）
initial_guesses = {
    'L1': [0.5, 0.0], 'L2': [1.1, 0.0], 'L3': [-1.1, 0.0],
    'L4': [0.48785, +np.sqrt(3)/2], 'L5': [0.48785, -np.sqrt(3)/2]
}

# 数值求解 + 误差分析
print("拉格朗日点数值坐标与理论解对比（单位：无量纲）")
print("-" * 60)
L_points = {}
for name, guess in initial_guesses.items():
    sol = root(grad_phi, guess, args=(mu,), method='lm', tol=eps)
    L_points[name] = sol.x
    theo = np.array(L_theory[name])
    error = np.linalg.norm(sol.x - theo)
    print(f"{name}: 数值 = {sol.x}, 理论 = {theo}, 误差 = {error:.2e}")

# 模块二：运动方程与轨道模拟
# 参数设置（地月系统）
mu = 0.04
mu1, mu2 = 1 - mu, mu

# 拉格朗日点
L_points = {
    'L1': [0.83691491, 0.0],
    'L2': [1.15567999, 0.0],
    'L3': [-1.00506003, 0.0],
    'L4': [0.4877, 0.8660254],
    'L5': [0.4877, -0.8660254],
}

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

# 模块三：李雅普诺夫指数估算
def rt_deriv(t, state, mu):
    """
    限制性三体问题在旋转坐标系下的运动微分方程
    参数:
        t: 时间（这里并不影响方程，但 solve_ivp 需要）
        state: 状态向量 [x, y, vx, vy]
        mu: 质量比
    返回:
        dstate_dt: 导数向量 [vx, vy, ax, ay]
    """
    x, y, vx, vy = state
    # 主星与伴星的位置
    x1, x2 = -mu, 1 - mu

    # 距离
    r1 = np.sqrt((x + mu)**2 + y**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2)

    # 加速度
    ax = 2*vy + x - (1 - mu)*(x + mu)/r1**3 - mu*(x - 1 + mu)/r2**3
    ay = -2*vx + y - (1 - mu)*y/r1**3 - mu*y/r2**3

    return [vx, vy, ax, ay]

def estimate_max_lyapunov(mu, y0, dt=0.1, T=100, d0=1e-6):
    from numpy.linalg import norm
    steps = int(T/dt)
    delta_y = np.array([d0, 0, 0, 0])
    y = np.array(y0)
    y_perturbed = y + delta_y

    lyapunov_exponents = []
    t_vals = []

    for i in range(steps):
        sol = solve_ivp(rt_deriv, [0, dt], y, args=(mu,), t_eval=[dt])
        sol_perturbed = solve_ivp(rt_deriv, [0, dt], y_perturbed, args=(mu,), t_eval=[dt])
        
        y = sol.y[:-1]
        y_perturbed = sol_perturbed.y[:, -1]

        delta_y = y_perturbed - y
        d = norm(delta_y)
        lyap = np.log(d / d0) / dt
        lyapunov_exponents.append(lyap)
        t_vals.append((i+1)*dt)

        # 重正化
        delta_y = d0 * delta_y / d
        y_perturbed = y + delta_y

    return np.array(t_vals), np.array(lyapunov_exponents)
