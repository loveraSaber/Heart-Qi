import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class Loss_Function(nn.Module): 
    def __init__(self, config, device):
        super(Loss_Function, self).__init__()
        self.device = device
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, label_num, label_num_prediction, label_score, label_score_prediction):

        if isinstance(label_num_prediction, list):
            label_num_prediction = torch.tensor(label_num_prediction, dtype=torch.float32, device=self.device)

        if isinstance(label_score_prediction, list):
            label_score_prediction = torch.tensor(label_score_prediction, dtype=torch.float32, device=self.device)

        loss_score_mse = self.mse_loss(label_score_prediction, label_score)
        # loss_score_mae = self.mae_loss(label_score_prediction, label_score)

        loss_class = self.mse_loss(label_num_prediction, label_num)
        total_loss = loss_score_mse + loss_class

        return {
            "total_loss": total_loss,
            "score_mse": loss_score_mse,
            "class_loss": loss_class
        }

