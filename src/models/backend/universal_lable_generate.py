import torch
import torch.nn as nn
from collections import OrderedDict
from ..backend.scale import STAI_Prediction, PHQ_Prediction, CPSS_Prediction, BPS_Prediction, FFMQ_Prediction, AIS_Prediction, TFEQ_Prediction, BFI_Prediction

class LabelGenerate(nn.Module):
    """通用的多模态独立预测模型——每个模态单独预测标签得分。"""
    def __init__(self, scale_type, input_dim):
        super(LabelGenerate, self).__init__()
        self.scale_type = scale_type
        if scale_type == "STAI":
            self.predict = STAI_Prediction(input_dim)
        elif scale_type == "PHQ-9":
            self.predict = PHQ_Prediction(input_dim)
        elif scale_type == "CPSS":
            self.predict = CPSS_Prediction(input_dim)
        elif scale_type == "BPS":
            self.predict = BPS_Prediction(input_dim)
        elif scale_type == "FFMQ":
            self.predict = FFMQ_Prediction(input_dim)
        elif scale_type == "AIS":
            self.predict = AIS_Prediction(input_dim)
        elif scale_type == "TFEQ":
            self.predict = TFEQ_Prediction(input_dim)
        elif scale_type == "BFI":
            self.predict = BFI_Prediction(input_dim)
            
    def forward(self, data):
        #检测data是不是tuple数据类型
        if isinstance(data, tuple):
            data = data[0]
        if self.scale_type == "STAI":
            result = self.predict(data)
        elif self.scale_type == "PHQ-9":
            result = self.predict(data)
        elif self.scale_type == "CPSS":
            result = self.predict(data)
        elif self.scale_type == "BPS":
            result = self.predict(data)
        elif self.scale_type == "FFMQ":
            result = self.predict(data)
        elif self.scale_type == "AIS":
            result = self.predict(data)
        elif self.scale_type == "TFEQ":
            result = self.predict(data)
        elif self.scale_type == "BFI":
            result = self.predict(data)
        return result
