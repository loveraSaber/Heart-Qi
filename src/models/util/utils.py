import os
import yaml
import torch
import numpy as np


"""--------------------------------------------------------填充--------------------------------------------------------"""

def pad_1D(inputs, max_len=None, PAD=0):
    # input: (batch_size, T1)
    def pad_data(x, length, PAD):
        if np.shape(x)[0] > max_len:
            print(np.shape(x))
            print(max_len)
            raise ValueError("not max_len")
            
            
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded
        
    if max_len:
        output = np.stack([pad_data(x, max_len, PAD) for x in inputs])
    else:
        max_len = max((len(x) for x in inputs))
        output = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return output


def pad_2D(inputs, maxlen=None):
    PAD=0.0
    def pad(x, max_mel_len, PAD):
        # 获取当前输入序列的 (length, dim)
        length, dim = x.shape
        # 如果当前序列的长度大于最大长度，抛出错误
        if length > max_mel_len:
            print("-"*80)
            print(f"length： {length}")
            print(f"max_mel_len： {max_mel_len}")
            print("-"*80)
            raise ValueError("Sequence length exceeds max_mel_len")
        
        # 对时间维度进行填充
        x_padded = np.pad(
            x, ((0, max_mel_len - length), (0, 0)), mode="constant", constant_values=PAD
        )
        return x_padded

    # 如果提供了 maxlen，使用该最大长度进行填充
    if maxlen:
        # 按批次处理每个序列
        output = np.array([pad(x, maxlen, PAD) for x in inputs])
    else:
        max_mel_len = max([input.size(0) for input in inputs])
        output = np.array([pad(x, max_mel_len, PAD) for x in inputs])

    return output


def pad_2D_tesor(inputs, maxlen=None, device=None, dim=80):
    pad_value=0.0
    batch = len(inputs)
    
    # 如果未指定 maxlen，计算所有序列的最大长度
    if maxlen is None:
        maxlen = max(x.shape[0] for x in inputs)
    
    # 初始化填充后的张量
    padded_inputs = torch.full((batch, maxlen, dim), pad_value, device=device)
    
    for i, x in enumerate(inputs):
        seq_len = x.shape[0]
        if seq_len > maxlen:
            raise ValueError(f"Sequence {i} exceeds maxlen ({seq_len} > {maxlen}).")
        padded_inputs[i, :seq_len, :] = x  # 填充当前序列
    padded_inputs = padded_inputs.to(device)
    return padded_inputs

"""--------------------------------------------------------Get--------------------------------------------------------"""


# 获取模型参数量
def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


# 通过梅尔谱图长度找mask
def get_mask(mel_lens, max_mel_len=None):
    batch_size = len(mel_lens)
    
    # 如果没有传入 max_mel_len，使用 mel_lens 中的最大长度
    if max_mel_len is None:
        if torch.is_tensor(mel_lens):
            max_mel_len = mel_lens.max().item()
        else:
            max_mel_len = max(mel_lens)
    
    # 初始化掩码张量，使用 torch.float32 更为安全
    mask = torch.zeros((batch_size, max_mel_len), dtype=torch.float32)
    
    # 根据 mel_lens 设置掩码为 1（即实际的有效部分）
    for i in range(batch_size):
        mask[i, :mel_lens[i]] = 1  # 设置实际数据部分为 1
    
    return mask


"""--------------------------------------------------------进模型前的处理--------------------------------------------------------"""

def to_device(data, device):
    
    # train
    if len(data) == 12:
        (basename, mels_pad, mel_lens, max_mel_len, videos_pad, video_lens, max_video_len, phonemes_pad, phoneme_lens, max_phoneme_len, labels, labels_score) = data
        
        mels = torch.from_numpy(mels_pad).float().to(device)
        mel_lens = torch.from_numpy(mel_lens).to(device)
        mel_mask = get_mask(mel_lens, max_mel_len).to(device)    
        
        videos = torch.from_numpy(videos_pad).float().to(device)
        video_lens = torch.from_numpy(video_lens).to(device)
        videos_mask = get_mask(video_lens, max_video_len).to(device)    
        
        phonemes = torch.from_numpy(phonemes_pad).long().to(device)
        phoneme_lens = torch.from_numpy(phoneme_lens).to(device)
        phoneme_mask = get_mask(phoneme_lens, max_phoneme_len).to(device)
        
        labels = torch.tensor(labels).long().to(device)
        labels_score = torch.tensor(labels_score).long().to(device)

        return (
            basename,
            mels,
            mel_lens,
            mel_mask,
            videos,
            video_lens,
            videos_mask,
            phonemes,
            phoneme_lens,
            phoneme_mask,
            labels,
            labels_score
        )

    # val
    if len(data) == 10:
        (basename, mels_pad, mel_lens, max_mel_len, videos_pad, video_lens, max_video_len, phonemes_pad, phoneme_lens, max_phoneme_len) = data
        
        mels = torch.from_numpy(mels_pad).float().to(device)
        mel_lens = torch.from_numpy(mel_lens).to(device)
        mel_mask = get_mask(mel_lens, max_mel_len).to(device)    
        
        videos = torch.from_numpy(videos_pad).float().to(device)
        video_lens = torch.from_numpy(video_lens).to(device)
        videos_mask = get_mask(video_lens, max_video_len).to(device)    
        
        phonemes = torch.from_numpy(phonemes_pad).long().to(device)
        phoneme_lens = torch.from_numpy(phoneme_lens).to(device)
        phoneme_mask = get_mask(phoneme_lens, max_phoneme_len).to(device)
        
        return (
            basename,
            mels,
            mel_lens,
            mel_mask,
            videos,
            video_lens,
            videos_mask,
            phonemes,
            phoneme_lens,
            phoneme_mask
        )


def ensure_dir(path):
    """
    确保路径的父目录（文件）或自身（目录）存在。
    支持以下情况：
    - 文件路径（如 "/path/to/file.txt"）→ 创建 "/path/to/"
    - 目录路径（如 "/path/to/dir/"）→ 创建 "/path/to/dir/"
    - 目录路径（如 "/path/to/dir"）→ 创建 "/path/to/dir/"
    """
    if os.path.splitext(path)[1]:  # 如果有扩展名，视为文件路径
        dir_path = os.path.dirname(path)
    else:  # 否则视为目录路径
        dir_path = path.rstrip("/")  # 移除末尾的斜杠（如果有）
    
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        
    
def log_and_save(loss,current_step,total_step,config,model,optimizer,logger):
    log_step = config["step"]["log_step"]            
    save_step = config["step"]["save_step"] 
    log_path = config["path"]["log_path"]
    save_path = config["path"]["save_path"]
    scale_type = config["scale_type"]   
     
    if current_step % log_step == 0:
        if scale_type == "AIS":
            total_loss = loss["total_loss"]
            label_1_loss = loss["label_1_loss"]
            label_2_loss = loss["label_2_loss"]
            label_3_loss = loss["label_3_loss"]
            label_4_loss = loss["label_4_loss"]
            
            label_1_acc = loss["label_1_acc"]
            label_2_acc = loss["label_2_acc"]
            label_3_acc = loss["label_3_acc"]
            label_4_acc = loss["label_4_acc"]

            message = f"{current_step}/{total_step} total: {total_loss:.4f}  l_1: {label_1_loss:.4f}  l_2: {label_2_loss:.4f} l_3: {label_3_loss:.4f} l_4: {label_4_loss:.4f} a_1: {label_1_acc:.4f}  a_2: {label_2_acc:.4f} a_3: {label_3_acc:.4f} a_4: {label_4_acc:.4f}"
            with open(os.path.join(log_path, "log_total.txt"), "a") as f:
                f.write(message + "\n")
                
            logger.add_scalar("Total", total_loss, current_step)
            logger.add_scalar("label_1_loss", label_1_loss, current_step)
            logger.add_scalar("label_2_loss", label_2_loss, current_step)
            logger.add_scalar("label_3_loss", label_1_loss, current_step)
            logger.add_scalar("label_4_loss", label_1_loss, current_step)
            
            logger.add_scalar("label_1_acc", label_1_acc, current_step)
            logger.add_scalar("label_1_acc", label_2_acc, current_step)
            logger.add_scalar("label_1_acc", label_3_acc, current_step)
            logger.add_scalar("label_1_acc", label_4_acc, current_step)
    
    if current_step % save_step == 0:
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            os.path.join(save_path, f"{current_step}.pth")
        )
        
        
def load_config(config_path, override_paths=None):

    # 1. 加载原始配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 2. 动态覆盖路径
    if override_paths:
        config['path'].update(override_paths)

    # 3. 确保所有路径存在
    for path_key in ['ckpt_path', 'log_path']:
        path_value = config['path'].get(path_key)
        ensure_dir(path_value)
            
    return config





