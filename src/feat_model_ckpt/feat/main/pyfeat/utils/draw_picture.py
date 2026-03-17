import numpy as np
from matplotlib import font_manager
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False     # 解决负号显示问题

def draw_emotion_radar(values, path):
    """
    绘制六维情感倾向雷达图
    参数：
        values: np.array, shape (6, )
    """
    # 六维情感倾向（0–1，彼此独立）
    labels = ["愤怒", "厌恶", "恐惧", "快乐", "悲伤", "惊讶"]

    # 雷达角度
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])
    values = np.concatenate([values, [values[0]]])

    # 绘图
    plt.figure(figsize=(9, 9))
    ax = plt.subplot(111, polar=True)

    ax.plot(angles, values, linewidth=3)
    ax.fill(angles, values, alpha=0.35)

    ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels, fontsize=13)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=11)

    ax.set_title("Six-Emotion Tendency Radar", fontsize=18, pad=25)

    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()