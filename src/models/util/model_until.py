import os
import torch
from rl import ScheduledOptim
from models.backbone import ScaleRecognition, SingleScaleRecognition

def get_model(config, device, train=False, restore_step=None, ckpt_path=None):

    # 初始化模型
    if config["model"] == "multi_model":
        model = ScaleRecognition(config, device)
    elif config["model"] == "single_model":
        model = SingleScaleRecognition(config, device)
    
    scheduled_optim = None  
    # 检查点加载逻辑
    if restore_step or ckpt_path:
        # 如果未直接指定ckpt_path，则根据restore_step构造路径
        if ckpt_path is None:
            ckpt_path = os.path.join(
                config["path"]["ckpt_path"],
                f"{restore_step}.pth"
            )
        
        # 检查检查点文件是否存在
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"未找到模型检查点文件: {ckpt_path}")
        
        print(f"正在从以下路径加载模型: {ckpt_path}")
        # 加载检查点(自动映射到指定设备)
        ckpt = torch.load(ckpt_path, map_location=device)
        
        model.load_state_dict(ckpt["model"])
    
    # 训练模式设置
    if train:

        scheduled_optim = ScheduledOptim(model, config, restore_step or 0)
        
        # 如果从检查点恢复且包含优化器状态
        if restore_step and "optimizer" in ckpt:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        
        model.train()  # 设置为训练模式
    else:
        # 推理模式设置
        model.eval()
        model.requires_grad_(False)  # 禁用梯度计算
    
    return model, scheduled_optim



