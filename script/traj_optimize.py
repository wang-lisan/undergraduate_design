import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt


# =========================
# 工具函数：角度展开/回绕
# =========================
def unwrap_angles(a):
    """把角度序列做 unwrap，避免跨越 ±pi 时插值出现跳变（比如从 +3.13 跳到 -3.14）"""
    return np.unwrap(a)


def wrap_to_pi(a):
    """把角度包回 [-pi, pi]，便于输出更“正常”"""
    return (a + np.pi) % (2 * np.pi) - np.pi


# =========================
# 工具函数：巴特沃斯低通 + 零相位滤波
# =========================
def lowpass_filtfilt(x, fs, cutoff_hz=6.0, order=4):
    """
    x: (N,) 或 (N, D) 的数据
    fs: 采样率
    cutoff_hz: 截止频率
    order: 滤波器阶数
    """
    nyq = 0.5 * fs
    wn = cutoff_hz / nyq
    b, a = butter(order, wn, btype="low")
    return filtfilt(b, a, x, axis=0)


# =========================
# 主流程
# =========================
def main():
    # ---------
    # 1) 输入输出文件名
    # ---------
    raw_file = "raw_traj.csv"
    new_file = "new_traj.csv"

    # 打印当前工作目录，避免你又找不到文件
    cwd = os.getcwd()
    print("当前工作目录：", cwd)

    # 检查原始文件是否存在
    if not os.path.exists(raw_file):
        raise FileNotFoundError(
            f"找不到 {raw_file}。\n"
            f"请确认 {raw_file} 和这个脚本在同一文件夹，或把 raw_file 改成绝对路径。\n"
            f"当前工作目录是：{cwd}"
        )

    # ---------
    # 2) 读取 raw_traj.csv
    # ---------
    raw = pd.read_csv(raw_file)

    # 必要列检查（与你之前生成的格式匹配）
    required = ["time_s", "x_m", "y_m", "z_m", "roll_rad", "pitch_rad", "yaw_rad"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"{raw_file} 缺少列：{missing}")

    # 按时间排序，去除重复时间点（重复会导致 interp1d 报错）
    raw = raw.sort_values("time_s").drop_duplicates("time_s", keep="first").reset_index(drop=True)

    t_raw = raw["time_s"].to_numpy()
    xyz_raw = raw[["x_m", "y_m", "z_m"]].to_numpy()
    rpy_raw = raw[["roll_rad", "pitch_rad", "yaw_rad"]].to_numpy()

    # ---------
    # 3) 构造 100Hz 均匀时间轴
    # ---------
    fs_new = 100.0
    dt_new = 1.0 / fs_new
    t0, t1 = float(t_raw[0]), float(t_raw[-1])
    t_new = np.arange(t0, t1 + 1e-12, dt_new)

    # ---------
    # 4) 插值到 100Hz
    #    位置：直接线性插值
    #    姿态：先对每个角 unwrap，再线性插值（否则会跨 pi 跳变）
    # ---------
    # 位置插值
    fx = interp1d(t_raw, xyz_raw[:, 0], kind="linear", fill_value="extrapolate")
    fy = interp1d(t_raw, xyz_raw[:, 1], kind="linear", fill_value="extrapolate")
    fz = interp1d(t_raw, xyz_raw[:, 2], kind="linear", fill_value="extrapolate")
    xyz_i = np.column_stack([fx(t_new), fy(t_new), fz(t_new)])

    # 姿态插值（unwrap）
    r_un = unwrap_angles(rpy_raw[:, 0])
    p_un = unwrap_angles(rpy_raw[:, 1])
    y_un = unwrap_angles(rpy_raw[:, 2])

    fr = interp1d(t_raw, r_un, kind="linear", fill_value="extrapolate")
    fp = interp1d(t_raw, p_un, kind="linear", fill_value="extrapolate")
    fyaw = interp1d(t_raw, y_un, kind="linear", fill_value="extrapolate")
    rpy_i = np.column_stack([fr(t_new), fp(t_new), fyaw(t_new)])

    # ---------
    # 5) 低通滤波（对插值后的数据滤波）
    #    cutoff 可以按你的噪声强度/动作快慢调：
    #    - 动作慢：4~6 Hz
    #    - 动作稍快：6~10 Hz
    # ---------
    cutoff_pos = 6.0
    cutoff_ang = 6.0

    xyz_f = lowpass_filtfilt(xyz_i, fs=fs_new, cutoff_hz=cutoff_pos, order=4)
    rpy_f = lowpass_filtfilt(rpy_i, fs=fs_new, cutoff_hz=cutoff_ang, order=4)

    # 回绕到 [-pi, pi]（可选，但通常输出更舒服）
    rpy_f = np.column_stack([
        wrap_to_pi(rpy_f[:, 0]),
        wrap_to_pi(rpy_f[:, 1]),
        wrap_to_pi(rpy_f[:, 2]),
    ])

    # ---------
    # 6) 保存 new_traj.csv
    # ---------
    out = pd.DataFrame({
        "time_s": t_new,
        "x_m": xyz_f[:, 0],
        "y_m": xyz_f[:, 1],
        "z_m": xyz_f[:, 2],
        "roll_rad": rpy_f[:, 0],
        "pitch_rad": rpy_f[:, 1],
        "yaw_rad": rpy_f[:, 2],
    })
    out.to_csv(new_file, index=False)
    print("已保存：", os.path.abspath(new_file))

    # ---------
    # 7) 画图对比：XY 平面 + 3D
    # ---------
    plt.figure()
    plt.plot(xyz_raw[:, 0], xyz_raw[:, 1], label="raw (irregular, noisy)")
    plt.plot(xyz_f[:, 0], xyz_f[:, 1], label="new (100Hz + lowpass)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Trajectory Compare (XY)")
    plt.axis("equal")
    plt.legend()
    plt.show()


    # ============
    # 额外新增：分开看两条线（保留原来的叠加图不变）
    # ============

    # 1) 单独看 raw（XY）
    plt.figure()
    plt.plot(xyz_raw[:, 0], xyz_raw[:, 1])
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Raw Trajectory (XY)")
    plt.axis("equal")
    plt.show()

    # 2) 单独看 new（XY）
    plt.figure()
    plt.plot(xyz_f[:, 0], xyz_f[:, 1])
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("New Trajectory (100Hz + Lowpass) (XY)")
    plt.axis("equal")
    plt.show()

    
    # 3D 对比（可选）
    try:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(xyz_raw[:, 0], xyz_raw[:, 1], xyz_raw[:, 2], label="raw")
        ax.plot(xyz_f[:, 0], xyz_f[:, 1], xyz_f[:, 2], label="new")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.set_title("Trajectory Compare (3D)")
        ax.legend()
        plt.show()
    except Exception as e:
        print("3D 图跳过：", e)

    # 打印 dt 统计，验证“raw 不均匀、new 均匀 100Hz”
    dt_raw = np.diff(t_raw)
    dt_new_arr = np.diff(t_new)
    print("\n=== 时间间隔统计 ===")
    print("raw: mean=%.6f s, std=%.6f s, min=%.6f s, max=%.6f s"
          % (dt_raw.mean(), dt_raw.std(), dt_raw.min(), dt_raw.max()))
    print("new: mean=%.6f s, std=%.6f s, min=%.6f s, max=%.6f s"
          % (dt_new_arr.mean(), dt_new_arr.std(), dt_new_arr.min(), dt_new_arr.max()))


if __name__ == "__main__":
    main()
