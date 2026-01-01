import numpy as np
import pandas as pd

# ============================================================
# 相关噪声（低频漂移）：AR(1) 模型
# x[t] = alpha*x[t-1] + eps[t]
# alpha 越接近 1，越像慢慢飘的传感器漂移/手抖低频
# ============================================================
def ar1_noise(n, sigma=1.0, alpha=0.98, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    eps = rng.normal(0.0, sigma, size=n)
    x = np.zeros(n, dtype=float)
    for i in range(1, n):
        x[i] = alpha * x[i - 1] + eps[i]
    x -= np.mean(x)  # 去均值，避免整体偏移过大
    return x

# ============================================================
# 生成“非均匀”的时间戳（单调递增）
# 思路：
# 1) 用平均采样间隔 dt_mean = 1/fs
# 2) 在每个 dt 上加一个很小的抖动 jitter（正态分布）
# 3) 把 dt 限幅：避免出现负间隔或突然很大
# 4) 累加得到 time_s
# 5) 可选：让最后一个时间点对齐到 duration（整体不漂）
# ============================================================
def generate_irregular_timestamps(
    duration_s=12.0,
    fs_hz=200.0,
    jitter_std_ratio=0.08,  # 抖动强度占 dt 的比例：0.08 表示 dt 的 8% 标准差
    dt_min_ratio=0.5,       # 最小 dt = dt_mean * 0.5（防止负 dt 或过小）
    dt_max_ratio=1.8,       # 最大 dt = dt_mean * 1.8（防止突然拉长）
    seed=7
):
    rng = np.random.default_rng(seed)
    dt_mean = 1.0 / fs_hz

    # 预估需要多少个点（多给一点，后面按 duration 截断）
    n_guess = int(duration_s * fs_hz * 1.2) + 50

    # 为每个采样间隔生成抖动
    jitter = rng.normal(0.0, dt_mean * jitter_std_ratio, size=n_guess)

    # 得到非均匀 dt，并进行限幅
    dt = dt_mean + jitter
    dt = np.clip(dt, dt_mean * dt_min_ratio, dt_mean * dt_max_ratio)

    # 累加得到时间轴（从 0 开始）
    t = np.cumsum(dt)
    t = np.insert(t, 0, 0.0)  # 把起点 t=0 加进去

    # 截断到 duration 范围内
    t = t[t <= duration_s]

    # 为了让“整体时长”更贴近 duration，可把末端对齐（不会改变相对不均匀特性太多）
    if len(t) >= 2 and t[-1] > 1e-9:
        t = t * (duration_s / t[-1])

    return t

# ============================================================
# 生成八字轨迹（∞）+ xyz + rpy + 噪声 + 非均匀时间戳
# 输出：DataFrame
# ============================================================
def generate_figure8_xyz_rpy_irregular_time(
    duration_s=12.0,
    fs_hz=200.0,
    # 时间戳抖动参数
    jitter_std_ratio=0.08,
    dt_min_ratio=0.5,
    dt_max_ratio=1.8,
    # 八字形状参数
    scale_xy=0.25,     # 米
    z_base=0.30,       # 米
    z_amp=0.05,        # 米
    speed_hz=0.22,     # 八字循环频率（Hz）
    # 姿态变化（手腕感）
    yaw_amp=0.60,      # rad
    pitch_amp=0.35,    # rad
    roll_amp=0.25,     # rad
    # 位置噪声（白噪声 + 漂移）
    pos_white_sigma=0.0015,
    pos_drift_sigma=0.0002,
    pos_drift_alpha=0.995,
    # 姿态噪声（白噪声 + 漂移）
    ang_white_sigma=0.008,
    ang_drift_sigma=0.001,
    ang_drift_alpha=0.995,
    seed=7
):
    rng = np.random.default_rng(seed)

    # ----------------------------
    # 1) 生成非均匀时间戳
    # ----------------------------
    t = generate_irregular_timestamps(
        duration_s=duration_s,
        fs_hz=fs_hz,
        jitter_std_ratio=jitter_std_ratio,
        dt_min_ratio=dt_min_ratio,
        dt_max_ratio=dt_max_ratio,
        seed=seed
    )
    n = len(t)

    # ----------------------------
    # 2) 八字位置轨迹：Lemniscate of Gerono（∞）
    # x = a*sin(wt)
    # y = a*sin(wt)*cos(wt)
    # z = z_base + z_amp*sin(2wt + phase)
    # 注意：现在 t 不是等间隔，但公式照样成立
    # ----------------------------
    w = 2.0 * np.pi * speed_hz
    x = scale_xy * np.sin(w * t)
    y = scale_xy * np.sin(w * t) * np.cos(w * t)
    z = z_base + z_amp * np.sin(2.0 * w * t + np.pi / 6.0)

    # ----------------------------
    # 3) 姿态（RPY）随时间平滑变化
    # 这里用“yaw->pitch->roll”节奏不同的正弦，模拟手腕自然摆动
    # 输出字段顺序按：roll, pitch, yaw
    # ----------------------------
    yaw = yaw_amp * np.sin(w * t + 0.3)
    pitch = pitch_amp * np.sin(2.0 * w * t + 1.1)
    roll = roll_amp * np.sin(1.5 * w * t + 2.0)

    # ----------------------------
    # 4) 叠加位置噪声：白噪声 + 低频漂移
    # ----------------------------
    pos_white = rng.normal(0.0, pos_white_sigma, size=(n, 3))
    pos_drift = np.column_stack([
        ar1_noise(n, sigma=pos_drift_sigma, alpha=pos_drift_alpha, rng=rng),
        ar1_noise(n, sigma=pos_drift_sigma, alpha=pos_drift_alpha, rng=rng),
        ar1_noise(n, sigma=pos_drift_sigma, alpha=pos_drift_alpha, rng=rng),
    ])

    x_n = x + pos_white[:, 0] + pos_drift[:, 0]
    y_n = y + pos_white[:, 1] + pos_drift[:, 1]
    z_n = z + pos_white[:, 2] + pos_drift[:, 2]

    # ----------------------------
    # 5) 叠加姿态噪声：白噪声 + 低频漂移
    # ----------------------------
    ang_white = rng.normal(0.0, ang_white_sigma, size=(n, 3))
    ang_drift = np.column_stack([
        ar1_noise(n, sigma=ang_drift_sigma, alpha=ang_drift_alpha, rng=rng),
        ar1_noise(n, sigma=ang_drift_sigma, alpha=ang_drift_alpha, rng=rng),
        ar1_noise(n, sigma=ang_drift_sigma, alpha=ang_drift_alpha, rng=rng),
    ])

    # 按 [roll, pitch, yaw] 通道分别加噪声
    roll_n = roll + ang_white[:, 0] + ang_drift[:, 0]
    pitch_n = pitch + ang_white[:, 1] + ang_drift[:, 1]
    yaw_n = yaw + ang_white[:, 2] + ang_drift[:, 2]

    # ----------------------------
    # 6) 打包成表格
    # ----------------------------
    df = pd.DataFrame({
        "time_s": t,
        "x_m": x_n,
        "y_m": y_n,
        "z_m": z_n,
        "roll_rad": roll_n,
        "pitch_rad": pitch_n,
        "yaw_rad": yaw_n,
    })

    return df

# ============================================================
# 入口：生成 CSV
# ============================================================
def main():
    df = generate_figure8_xyz_rpy_irregular_time(
        duration_s=12.0,
        fs_hz=200.0,
        jitter_std_ratio=0.08,  # 时间戳抖动强度（建议 0.03~0.15）
        dt_min_ratio=0.6,
        dt_max_ratio=1.6,
        seed=7
    )

    out_csv = "raw_traj.csv"
    df.to_csv(out_csv, index=False)
    print("已保存：", out_csv)

    # 打印一些信息，方便你确认“时间戳确实不均匀但不夸张”
    dt = np.diff(df["time_s"].to_numpy())
    print("样本数：", len(df))
    print("dt 统计：mean=%.6f s, std=%.6f s, min=%.6f s, max=%.6f s"
          % (dt.mean(), dt.std(), dt.min(), dt.max()))
    print(df.head())

if __name__ == "__main__":
    main()
