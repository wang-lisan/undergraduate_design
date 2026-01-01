import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 工具函数：差分、角度处理
# =========================
def finite_diff(y: np.ndarray, dt: float):
    """
    用中心差分估计速度/加速度。
    y: (T, D)
    return: dy (T,D), ddy (T,D)
    """
    dy = np.zeros_like(y)
    ddy = np.zeros_like(y)

    dy[1:-1] = (y[2:] - y[:-2]) / (2.0 * dt)
    dy[0] = (y[1] - y[0]) / dt
    dy[-1] = (y[-1] - y[-2]) / dt

    ddy[1:-1] = (dy[2:] - dy[:-2]) / (2.0 * dt)
    ddy[0] = (dy[1] - dy[0]) / dt
    ddy[-1] = (dy[-1] - dy[-2]) / dt
    return dy, ddy


def wrap_to_pi(angle: np.ndarray):
    """把角度 wrap 到 [-pi, pi]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi


# =========================
# 离散 DMP（最小实现）
# =========================
class DiscreteDMP:
    """
    离散 DMP（Ijspeert 风格）

    canonical:
        x_dot = -a_x * x / tau

    transform:
        tau*z_dot = a_z*(b_z*(g - y) - z) + f(x)
        tau*y_dot = z

    forcing:
        f(x) = (sum w_i*psi_i / sum psi_i) * x * (g - y0)
    """

    def __init__(self, n_bfs=80, a_z=25.0, b_z=None, a_x=1.0, ridge=1e-8):
        self.n_bfs = int(n_bfs)
        self.a_z = float(a_z)
        self.b_z = float(a_z / 4.0) if b_z is None else float(b_z)
        self.a_x = float(a_x)
        self.ridge = float(ridge)

        # 训练后参数
        self.centers = None  # (n_bfs,)
        self.widths = None   # (n_bfs,)
        self.w = None        # (D, n_bfs)
        self.y0 = None       # (D,)
        self.g = None        # (D,)

    def _build_kernels(self):
        # 在 x 空间指数分布中心
        self.centers = np.exp(-self.a_x * np.linspace(0, 1, self.n_bfs))

        # 宽度与中心间距相关（经验）
        diffs = np.diff(self.centers, append=self.centers[-1] * 0.5)
        diffs = np.maximum(np.abs(diffs), 1e-6)
        self.widths = 1.0 / (diffs ** 2)

    def _psi(self, x: np.ndarray):
        """
        x: (T,)
        return psi: (T, n_bfs)
        """
        x = x.reshape(-1, 1)
        c = self.centers.reshape(1, -1)
        h = self.widths.reshape(1, -1)
        return np.exp(-h * (x - c) ** 2)

    def fit(self, y_demo: np.ndarray, dt: float):
        """
        y_demo: (T, D)
        dt: 采样周期
        """
        y_demo = np.asarray(y_demo, dtype=float)
        T, D = y_demo.shape

        self._build_kernels()

        self.y0 = y_demo[0].copy()
        self.g = y_demo[-1].copy()

        dy, ddy = finite_diff(y_demo, dt)
        tau = (T - 1) * dt

        # canonical x(t)
        x = np.zeros(T)
        x[0] = 1.0
        for t in range(1, T):
            x[t] = x[t - 1] + (-self.a_x * x[t - 1]) * dt / tau
        x = np.clip(x, 0.0, 1.0)

        # 反推 f_target
        # f_target = tau^2*y_ddot - a_z*(b_z*(g-y) - tau*y_dot)
        f_target = np.zeros((T, D))
        for d in range(D):
            f_target[:, d] = (tau**2) * ddy[:, d] - self.a_z * (self.b_z * (self.g[d] - y_demo[:, d]) - tau * dy[:, d])

        # 回归：(f_target / (g-y0)) ≈ Phi @ w
        eps = 1e-8
        scale = (self.g - self.y0)
        scale_safe = np.where(np.abs(scale) < eps, np.sign(scale) * eps + eps, scale)

        psi = self._psi(x)  # (T, n_bfs)
        psi_sum = np.sum(psi, axis=1, keepdims=True) + 1e-12
        Phi = (psi / psi_sum) * x.reshape(-1, 1)  # (T, n_bfs)

        # 岭回归解 w
        A = Phi.T @ Phi + self.ridge * np.eye(self.n_bfs)
        A_inv = np.linalg.inv(A)

        self.w = np.zeros((D, self.n_bfs))
        for d in range(D):
            y_reg = f_target[:, d] / scale_safe[d]
            self.w[d] = (A_inv @ (Phi.T @ y_reg)).ravel()

        return self

    def rollout(self, dt: float, timesteps: int, y0=None, g=None, tau=None):
        """
        生成轨迹
        dt: 采样周期
        timesteps: 点数
        y0/g: 新起点/终点（不传则用训练时）
        tau: 总时长（不传则用 timesteps*dt）
        """
        assert self.w is not None, "请先 fit()"

        D = self.w.shape[0]
        y0 = self.y0.copy() if y0 is None else np.asarray(y0, dtype=float).copy()
        g = self.g.copy() if g is None else np.asarray(g, dtype=float).copy()
        if tau is None:
            tau = (timesteps - 1) * dt

        y = np.zeros((timesteps, D))
        z = np.zeros((timesteps, D))  # z = tau*y_dot
        y[0] = y0

        x = 1.0
        for t in range(1, timesteps):
            # canonical
            x = x + (-self.a_x * x) * dt / tau
            x = float(np.clip(x, 0.0, 1.0))

            # basis
            psi = np.exp(-self.widths * (x - self.centers) ** 2)
            psi_sum = np.sum(psi) + 1e-12
            basis = (psi / psi_sum) * x  # (n_bfs,)

            # forcing
            scale = (g - y0)
            f = np.zeros(D)
            for d in range(D):
                f[d] = (basis @ self.w[d]) * scale[d]

            # transform system
            z_dot = (self.a_z * (self.b_z * (g - y[t - 1]) - z[t - 1]) + f) / tau
            z[t] = z[t - 1] + z_dot * dt

            y_dot = z[t] / tau
            y[t] = y[t - 1] + y_dot * dt

        return y


# =========================
# 画图函数：3D + 6轴随时间
# =========================
def plot_xyz_3d(traj_xyz, title, label=None):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(traj_xyz[:, 0], traj_xyz[:, 1], traj_xyz[:, 2], linewidth=2, label=label)
    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.grid(True)
    if label is not None:
        ax.legend()
    plt.tight_layout()
    return fig, ax


def plot_xyz_overlap(traj_a, traj_b, label_a, label_b, title):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(traj_a[:, 0], traj_a[:, 1], traj_a[:, 2], linewidth=2, label=label_a)
    ax.plot(traj_b[:, 0], traj_b[:, 1], traj_b[:, 2], linewidth=2, linestyle="--", label=label_b)
    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig, ax


def plot_6d_time(t, y_demo, y_pred, title):
    """
    画 6轴随时间对比：demo vs pred
    """
    names = ["x", "y", "z", "roll", "pitch", "yaw"]
    fig, axes = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
    for i in range(6):
        axes[i].plot(t, y_demo[:, i], label="Demo", linewidth=1.5)
        axes[i].plot(t, y_pred[:, i], label="DMP", linewidth=1.2, linestyle="--")
        axes[i].set_ylabel(names[i])
        axes[i].grid(True)
        if i == 0:
            axes[i].legend()
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(title)
    plt.tight_layout()
    return fig, axes


# =========================
# 主流程：读取 -> DMP -> 预测 -> 新起终点预测 -> 画图 -> 输出 CSV
# =========================
def main():
    # 1) 读取同目录的 new_traj.csv
    df = pd.read_csv("new_traj.csv")

    # 2) 取出 time + xyz+rpy（6轴）
    time_col = "time_s"
    cols_6d = ["x_m", "y_m", "z_m", "roll_rad", "pitch_rad", "yaw_rad"]
    assert all(c in df.columns for c in [time_col] + cols_6d), f"CSV列不匹配: {df.columns}"

    t = df[time_col].to_numpy()
    dt = float(np.median(np.diff(t)))
    Y = df[cols_6d].to_numpy()   # (T,6)

    # 角度 unwrap（更稳，即使你已处理也没坏处）
    Y_fit = Y.copy()
    for j in [3, 4, 5]:
        Y_fit[:, j] = np.unwrap(Y_fit[:, j])

    # 3) DMP 拟合
    dmp = DiscreteDMP(n_bfs=80, a_z=25.0, b_z=25.0/4.0, a_x=1.0, ridge=1e-8)
    dmp.fit(Y_fit, dt=dt)

    # 4) 原起终点预测（应接近原轨迹）
    Y_pred_same = dmp.rollout(dt=dt, timesteps=len(t))

    # 输出角度 wrap 回 [-pi, pi]
    Y_pred_same_out = Y_pred_same.copy()
    for j in [3, 4, 5]:
        Y_pred_same_out[:, j] = wrap_to_pi(Y_pred_same_out[:, j])

    # 保存 CSV（格式与输入一致）
    df_pred_same = df.copy()
    df_pred_same[cols_6d] = Y_pred_same_out
    df_pred_same.to_csv("dmp_pred_same_start_end.csv", index=False)

    # 5) 新起终点（略微不同）——你可以自行改成真实 A1/B1
    y0_new = Y_fit[0].copy()
    g_new = Y_fit[-1].copy()

    # 这里只对 xyz 做轻微偏移（rpy 保持不变）；你也可以一起偏移 rpy
    y0_new[0:3] += np.array([0.01, -0.005, 0.0])   # 起点偏移
    g_new[0:3]  += np.array([-0.01, 0.008, 0.0])   # 终点偏移

    Y_pred_new = dmp.rollout(dt=dt, timesteps=len(t), y0=y0_new, g=g_new)

    # wrap 角度输出
    Y_pred_new_out = Y_pred_new.copy()
    for j in [3, 4, 5]:
        Y_pred_new_out[:, j] = wrap_to_pi(Y_pred_new_out[:, j])

    # 保存 CSV（格式与输入一致）
    df_pred_new = df.copy()
    df_pred_new[cols_6d] = Y_pred_new_out
    df_pred_new.to_csv("dmp_pred_new_start_end.csv", index=False)

    # =========================
    # 6) 画图（3D xyz 三张 + 新起终点三张）
    # =========================
    xyz_demo = Y[:, 0:3]
    xyz_pred_same = Y_pred_same_out[:, 0:3]
    xyz_pred_new = Y_pred_new_out[:, 0:3]

    # 第一组：原起终点
    plot_xyz_3d(xyz_demo, "Demo trajectory (XYZ)")
    plot_xyz_3d(xyz_pred_same, "DMP trajectory (same start/end)")
    plot_xyz_overlap(xyz_demo, xyz_pred_same, "Demo", "DMP", "Overlap: Demo vs DMP (same start/end)")

    # 第二组：新起终点
    plot_xyz_3d(xyz_demo, "Demo trajectory (XYZ) - reference")
    plot_xyz_3d(xyz_pred_new, "DMP trajectory (new start/end)")
    plot_xyz_overlap(xyz_demo, xyz_pred_new, "Demo", "DMP new", "Overlap: Demo vs DMP (new start/end)")

    # 额外：6轴随时间对比（你没强制要求，但非常实用）
    plot_6d_time(t, Y, Y_pred_same_out, "6D compare: Demo vs DMP (same start/end)")
    plot_6d_time(t, Y, Y_pred_new_out, "6D compare: Demo vs DMP (new start/end)")

    plt.show()

    print("Done.")
    print(f"dt = {dt:.6f} s  (expected ~0.01 for 100Hz)")
    print("Saved: dmp_pred_same_start_end.csv")
    print("Saved: dmp_pred_new_start_end.csv")
    print("New y0 xyz =", y0_new[:3], "New g xyz =", g_new[:3])


if __name__ == "__main__":
    main()
