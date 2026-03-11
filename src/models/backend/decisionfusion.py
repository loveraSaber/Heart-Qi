import torch
import torch.nn.functional as F
from torch import nn

class LabelFusion(nn.Module):

    def __init__(self,config, device):
        super().__init__()
        self.device = device
        modal_num = config["modal_num"]

        self.mlp = nn.Sequential(
            nn.Linear(modal_num, modal_num),
            nn.GELU(),
            nn.Linear(modal_num, 1)
        ) 

    def forward(self, mel_score, video_score, phoneme_score):
        cat_data = torch.cat([mel_score, video_score, phoneme_score], dim=-1)
        prediction = self.mlp(cat_data)
        return prediction
    


class DecisionFusion(nn.Module):
    def __init__(self, num_modalities=3):
        super(DecisionFusion, self).__init__()
        self.num_modalities = num_modalities
        self.fusion = nn.Linear(num_modalities, 1)

    def forward(self, preds):
        """
        preds: list or tuple of modality predictions
               e.g., [pred_audio, pred_video, pred_text]
               each of shape [B, 1] or [B]
        """
        # 确保输入是 list/tuple
        if not isinstance(preds, (list, tuple)):
            raise TypeError("preds must be a list or tuple of tensors")


        # 拼接 [B, num_modalities]
        x = torch.cat(preds, dim=-1)

        # 线性融合
        fused = self.fusion(x)   # [B, 1]

        return fused

