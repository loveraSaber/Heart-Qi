import torch
import torch.nn.functional as F
from torch import nn
from ..backend.universal_lable_generate import LabelGenerate


class LabelPrediction(nn.Module):

    def __init__(self,config, scale_type, data_type):
        super().__init__()

        self.data_type = data_type
        
        if self.data_type == "mel":
            input_dim = config["mel"]["label_prediction"]["input_dim"]
            pred_dim = config["scale_feature"]
            
        elif self.data_type == "video":
            input_dim = config["video"]["label_prediction"]["input_dim"]
            pred_dim = config["scale_feature"]
            
        elif self.data_type == "phoneme":
            input_dim = config["phoneme"]["label_prediction"]["input_dim"]
        self.lable_generate = LabelGenerate(scale_type, input_dim)

    def forward(self, data):
        # print(self.data_type)
        prediction = self.lable_generate(data)
        return prediction
    

            
            
            
            
            


