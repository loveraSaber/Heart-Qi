from app.models.base import BaseModel
from src.models.single_modal_backbone import E2E
from app.config.config import Config
from app.utils.model_util import load_ckpt
from app.utils.file_util import load_mat
import torch
import numpy as np


class CPSSModel(BaseModel):

    def load(self, cfg_path: str, ckpt: str, device):
        # 读取参数
        cfg = Config(cfg_path).cfg
        self.device = device
        self.model = load_ckpt(cfg, ckpt, device)

    def predict(self, mat_path):
        # 读取参数
        video_features = load_mat(mat_path=mat_path)
        # 将numpy数据转换成tensor，并从cpu转移到gpu上
        video_features = torch.from_numpy(video_features.astype(np.float32)).to(self.device).unsqueeze(0)
        # 提取特征
        features = self.model.video_encoder(video_features, None)
        predict = self.model.label_prediction_video(features)
        predict = predict.squeeze(0).detach().cpu().numpy() * 56 + 14
        pressure = np.round(predict, 2)
        return pressure
