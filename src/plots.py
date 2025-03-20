import matplotlib.pyplot as plt
from collections import deque
import numpy as np


def plot_losses(losses, window=50):
    # 计算移动平均
    moving_averages = []
    window_values = deque(maxlen=window)

    for loss in losses:
        window_values.append(loss)
        moving_averages.append(np.mean(list(window_values)))

    # 创建图形
    fig = plt.figure(figsize=(12, 6))

    # 绘制原始损失值
    plt.plot(losses, alpha=0.3, color="blue", label="Raw Losses")

    # 绘制移动平均
    plt.plot(
        moving_averages,
        color="red",
        linewidth=2,
        label=f"Moving Average (window={window})",
    )

    plt.title("Training Losses")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.show()