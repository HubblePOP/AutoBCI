"""
步态脑电二分类标签生成脚本
Gait Phase Binary Label Generator

输入：运动捕捉数据中脚趾 Z 轴信号（TOE_z）
输出：每个采样点的 0/1 标签
  0 = 支撑期（stance）：脚趾在地面
  1 = 摆动期（swing）：脚趾离地

用法：
  python generate_gait_phase_labels.py

原理：
  1. 取 TOE_z 信号的中位数作为阈值
  2. 高于阈值 = 摆动（1），低于阈值 = 支撑（0）
  3. 短于 min_duration_ms 的区间视为噪声，合并到相邻状态

验证数据（walk_20240719_07）：
  - 前后肢步数比：99.9%
  - 摆动占比：~50%
  - 平均步态周期：~1.1s
"""

import numpy as np


def generate_gait_labels(
    signal: np.ndarray,
    sample_rate: float = 200.0,
    min_duration_ms: float = 120.0,
) -> np.ndarray:
    """
    从脚趾 Z 轴信号生成 0/1 步态标签。

    Parameters
    ----------
    signal : np.ndarray
        脚趾 Z 轴信号，shape = (n_samples,)
        Z 值高 = 脚离地（摆动），Z 值低 = 脚着地（支撑）
    sample_rate : float
        采样率，单位 Hz，默认 200
    min_duration_ms : float
        最短区间时长（毫秒），短于此值的区间会被合并，默认 120ms

    Returns
    -------
    labels : np.ndarray
        0/1 标签，shape = (n_samples,)
        0 = 支撑（stance），1 = 摆动（swing）
    """
    threshold = np.median(signal)
    labels = (signal > threshold).astype(np.int32)

    # 过滤短区间
    min_samples = int(min_duration_ms / 1000.0 * sample_rate)
    if min_samples < 2:
        return labels

    # 找到所有状态切换点
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(labels):
            # 找当前区间的起止
            val = labels[i]
            j = i + 1
            while j < len(labels) and labels[j] == val:
                j += 1
            # 如果区间太短，合并到周围
            if j - i < min_samples and i > 0 and j < len(labels):
                fill_val = labels[i - 1]  # 用前一个区间的值填充
                labels[i:j] = fill_val
                changed = True
            i = j

    return labels


def summarize_labels(labels: np.ndarray, sample_rate: float = 200.0) -> dict:
    """统计标签分布和步态参数。"""
    n = len(labels)
    n_stance = int(np.sum(labels == 0))
    n_swing = int(np.sum(labels == 1))

    # 找摆动区间
    swing_intervals = []
    stance_intervals = []
    i = 0
    while i < n:
        val = labels[i]
        j = i + 1
        while j < n and labels[j] == val:
            j += 1
        dur_ms = (j - i) / sample_rate * 1000
        if val == 1:
            swing_intervals.append(dur_ms)
        else:
            stance_intervals.append(dur_ms)
        i = j

    result = {
        "total_samples": n,
        "total_seconds": n / sample_rate,
        "stance_count": n_stance,
        "swing_count": n_swing,
        "stance_ratio": n_stance / n if n > 0 else 0,
        "swing_ratio": n_swing / n if n > 0 else 0,
        "num_swing_intervals": len(swing_intervals),
        "num_stance_intervals": len(stance_intervals),
    }

    if swing_intervals:
        result["avg_swing_ms"] = float(np.mean(swing_intervals))
        result["median_swing_ms"] = float(np.median(swing_intervals))

    if stance_intervals:
        result["avg_stance_ms"] = float(np.mean(stance_intervals))
        result["median_stance_ms"] = float(np.median(stance_intervals))

    if swing_intervals and stance_intervals:
        avg_cycle = np.mean(swing_intervals) + np.mean(stance_intervals)
        result["avg_cycle_ms"] = float(avg_cycle)
        result["step_freq_hz"] = float(1000.0 / avg_cycle) if avg_cycle > 0 else 0

    return result


# ── 示例用法 ──
if __name__ == "__main__":
    # 生成一段模拟信号做演示
    # 实际使用时替换为真实的 TOE_z 数据
    fs = 200.0
    duration = 10.0  # 10 秒
    t = np.arange(0, duration, 1.0 / fs)

    # 模拟步态信号：1Hz 的正弦波 + 噪声
    signal = np.sin(2 * np.pi * 1.0 * t) + 0.1 * np.random.randn(len(t))

    labels = generate_gait_labels(signal, sample_rate=fs, min_duration_ms=120)
    summary = summarize_labels(labels, sample_rate=fs)

    print("=== 标签生成结果 ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")

    print(f"\n标签分布: [stance, swing] = [{summary['stance_count']}, {summary['swing_count']}]")
    print(f"摆动占比: {summary['swing_ratio']:.1%}")

    # ── 实际数据使用方式 ──
    # import scipy.io  # 或者用你们自己的数据加载方式
    #
    # # 加载脚趾 Z 轴数据
    # toe_z = your_data["RHTOE_z"]  # shape = (n_samples,)
    # fs = 200.0  # 采样率
    #
    # # 生成标签
    # labels = generate_gait_labels(toe_z, sample_rate=fs, min_duration_ms=120)
    #
    # # 查看统计
    # summary = summarize_labels(labels, sample_rate=fs)
    # print(f"stance={summary['stance_count']}, swing={summary['swing_count']}")
    # print(f"摆动占比: {summary['swing_ratio']:.1%}")
    # print(f"步频: {summary['step_freq_hz']:.2f} Hz")
