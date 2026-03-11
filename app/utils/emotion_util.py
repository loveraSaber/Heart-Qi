import numpy as np


# 温控softmax
def softmax_with_temperature(z, T=2.0):
    z = z / T
    e = np.exp(z - z.max())
    return e / e.sum()

def emotion_tendency_pipeline(
    logits,            # shape: [T, 7] softmax output
    neutral_idx=0,           # neutral 在 softmax 中的 index
    baseline=0.05,           # 基础情感强度
    lambda_scale=5.0,        # 放大系数
    temperature=2.0                    # 温控参数
):
    """
    logits: np.array, shape (T, 7), T 帧数, 7 类情感 logits 情感顺序为 ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']
    neutral_idx: int, neutral 情感在 logits 中的索引位置
    baseline: float, 基础情感强度
    lambda_scale: float, 情感放大系数，值越大，情感倾向越明显
    temperature: float, softmax 温控参数
    return: 6维情感倾向向量, 每个值 ∈ [0,1]
    """
    # 判断输入数据是否符合要求
    T, C = logits.shape
    assert C == 7, "必须是7类情感模型"
    # 6维情感激活值
    basic_emotion = []
    neutrals = []
    # 计算每一帧的基础情感激活值
    for logit in logits:
        # 温控softmax函数
        p = softmax_with_temperature(logit, T=temperature)
        # 取出基础情感部分
        neutral = p[neutral_idx]
        # 记录 neutral 值以备后用
        neutrals.append(neutral)
        # 去除 neutral 后的基础情感概率分布
        p = np.delete(p, neutral_idx)
        # 对当前情感进行赋权
        a = p * (1 - neutral)
        basic_emotion.append(a)
    
    # 计算情感唤醒度
    emotion_arousal = 1 - np.array(neutrals).mean(axis=0)
    # 计算所有帧的基础情感激活值矩阵
    activations = np.array(basic_emotion).reshape(T, 6)
    # 时间平均
    a_avg = activations.mean(axis=0)

    # 加入基础情感权重
    b = np.full(6, baseline)
    
    s = a_avg + b
    # x = tanh(s)

    # 饱和映射（单调放大）
    S = 1 - np.exp(-lambda_scale * s)

    # Step 5: 裁剪确保数值合法
    S = np.clip(S, 0, 1)
    # 保留两位小数
    S = np.round(S, 2)
    emotion_arousal = np.round(emotion_arousal, 2)

    return S, emotion_arousal
