"""
可视化函数模块：用于绘制模拟结果和分析数据。
"""
import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numpy.linalg import norm

def compute_jacobi(state, mu):
    x, y, vx, vy = state
    r1 = np.sqrt((x + mu)**2 + y**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2)
    U = 0.5*(x**2 + y**2) + (1 - mu)/r1 + mu/r2
    return 2*U - (vx**2 + vy**2)

# 绘制轨道 + 零速度曲线
def plot_orbit(sol, point_name, show_zvc=True):
    plt.figure(figsize=(8, 6))
    
    # 轨道
    plt.plot(sol.y[0], sol.y[1], label=f"轨道 near {point_name}", color='blue')
    plt.scatter(*L_points[point_name], color='red', s=60, label=f"{point_name}")
    plt.scatter([-mu, 1 - mu], [0, 0], c='black', s=100, label='主星')

    # 零速度曲线（能量边界）
    # 绘制零速度曲线
    if show_zvc:
        try:
            vx0, vy0 = sol.y[2][0], sol.y[3][0]
            x0, y0 = sol.y[0][0], sol.y[1][0]
            E = 0.5 * (vx0**2 + vy0**2) - effective_potential(x0, y0, mu)
            X, Y = np.meshgrid(np.linspace(-1.5, 1.5, 400), np.linspace(-1.5, 1.5, 400))
            Phi = effective_potential(X, Y, mu)
            ZVC = E - Phi
            plt.contour(X, Y, ZVC, levels=[0], colors='gray', linestyles='--', linewidths=1)
        except Exception as e:
            print(f"[警告] 零速度曲线绘制失败：{e}")

    plt.xlabel("x"); plt.ylabel("y")
    plt.title(f"扰动轨道与零速度曲线 ({point_name})")
    plt.axis('equal'); plt.grid(True)
    plt.legend(); plt.tight_layout(); plt.show()

def plot_orbit(sol, point_name, C=None, show_zvc=True):
    fig, ax = plt.subplots(figsize=(8, 6))

    # 轨道轨迹
    ax.plot(sol.y[0], sol.y[1], label=f"轨道 near {point_name}", alpha=0.8)
    ax.scatter(*L_points[point_name], color='red', label=f"{point_name}", zorder=5)
    ax.scatter([-mu, 1 - mu], [0, 0], c='black', s=80, label='主星', zorder=5)

    # 零速度曲线（ZVC）
    if show_zvc:
        x = np.linspace(-1.5, 1.5, 500)
        y = np.linspace(-1.5, 1.5, 500)
        X, Y = np.meshgrid(x, y)
        R1 = np.sqrt((X + mu)**2 + Y**2)
        R2 = np.sqrt((X - 1 + mu)**2 + Y**2)
        U = 0.5 * (X**2 + Y**2) + (1 - mu)/R1 + mu/R2
        C_val = C if C is not None else compute_jacobi(sol.y[:, 0], mu)
        zvc = ax.contour(X, Y, 2*U, levels=[C_val], colors='gray', linewidths=1, linestyles='--')
        # # 为等高线设置标签（用于图例）
        # if zvc.collections:
        #     zvc.collections[0].set_label('零速度曲线（ZVC）')


    ax.set_title(f"{point_name}附近轨道")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal')
    plt.show()

# L1 与 L4 点扰动轨道稳定性对比
def plot_stability_comparison(mu):
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp

    fig, ax = plt.subplots(figsize=(8, 5))

    # 对 L1 和 L4 模拟轨道
    points = {
        'L1': {'delta': [0, 0, 0, 0.1], 'color': 'r'},
        'L4': {'delta': [0.01, 0, 0, 0], 'color': 'g'}
    }

    for label, config in points.items():
        sol = simulate_orbit(label, delta=config['delta'], t_max=100)
        x, y = sol.y[0], sol.y[1]
        r = np.sqrt(x**2 + y**2)
        ax.plot(sol.t, r, label=label, color=config['color'])

    ax.set_xlabel("时间 $t$")
    ax.set_ylabel("距离 $r(t)$")
    ax.set_title("L1 与 L4 点扰动轨道稳定性对比")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()
plot_stability_comparison(mu)
