import torch.nn as nn
from ..backend.decisionfusion import DecisionFusion
import torch

class Predict_Layer(nn.Module):
    def __init__(self, scale_type, num_modalities=3):
        super(Predict_Layer, self).__init__()
        self.scale_type = scale_type
        #三个数值可以计算loss，一个总结的需要预测 如何拆开
        self.num_modalities = num_modalities
        if num_modalities > 1:
            self.fusion = DecisionFusion(num_modalities) 
        if scale_type == "STAI":
            self.mse_loss = nn.MSELoss()
        elif scale_type == "PHQ-9":
            self.mse_loss = nn.MSELoss()
        elif scale_type == "CPSS":
            self.mse_loss = nn.MSELoss()
        elif self.scale_type == "BPS":
            self.mse_loss = nn.MSELoss()
        elif self.scale_type == "FFMQ":
            self.mse_loss = nn.MSELoss()
        elif self.scale_type == "AIS":
            self.mse_loss = nn.MSELoss()
            self.ce_loss = nn.CrossEntropyLoss()
        elif self.scale_type == "TFEQ":
            self.mse_loss = nn.MSELoss()
            self.ce_loss = nn.CrossEntropyLoss()
        elif self.scale_type == "BFI":
            self.mse_loss = nn.MSELoss()

    def forward(self, preds, label):
        label = label.float()
        mel_loss = video_loss = phoneme_loss = total_loss = 0.0
        results = []

        # -------- 多模态融合部分 --------
        if self.scale_type == "STAI":
            if self.num_modalities > 1:
            # 遍历每个模态
                for k, v in preds.items():
                    if v is not None:
                        results.append(v)

                if len(results) > 0:
                    total_pred = self.fusion(results)
                    total_loss = self.mse_loss(total_pred, label)
            # -------- 单模态损失部分 --------
            if preds.get("mel") is not None:
                mel_loss = self.mse_loss(preds["mel"] * 60, (label-20).squeeze(1))

            if preds.get("video") is not None:
                video_loss = self.mse_loss(preds["video"] * 60, (label-20).squeeze(1))

            if preds.get("phoneme") is not None:
                phoneme_loss = self.mse_loss(preds["phoneme"] * 60, label-20)

        elif self.scale_type == "PHQ-9": 
            label = (label-9).squeeze(1)
            if self.num_modalities > 1:
            # 遍历每个模态
                for k, v in preds.items():
                    if v is not None:
                        results.append(v)

                if len(results) > 0:
                    total_pred = self.fusion(results)
                    total_loss = self.mse_loss(total_pred, label)
            # -------- 单模态损失部分 --------
            if preds.get("mel") is not None:
                mel_loss = self.mse_loss(preds["mel"] * 27, label)
            if preds.get("video") is not None:
                video_loss = self.mse_loss(preds["video"] * 27, label)
            if preds.get("phoneme") is not None:
                phoneme_loss = self.mse_loss(preds["phoneme"] * 27, label)

        elif self.scale_type == "CPSS": 
            label = (label-14).squeeze(1)
            if self.num_modalities > 1:
            # 遍历每个模态
                for k, v in preds.items():
                    if v is not None:
                        results.append(v)

                if len(results) > 0:
                    total_pred = self.fusion(results)
                    total_loss = self.mse_loss(total_pred, label)
            # -------- 单模态损失部分 --------
            if preds.get("mel") is not None:
                mel_loss = self.mse_loss(preds["mel"] * 56, label)
            if preds.get("video") is not None:
                video_loss = self.mse_loss(preds["video"] * 56, label)
            if preds.get("phoneme") is not None:
                phoneme_loss = self.mse_loss(preds["phoneme"] * 56, label)

        elif self.scale_type == "BPS": 
            label = label[:,:,-4:]
            if self.num_modalities > 1:
            # 遍历每个模态
                for k, v in preds.items():
                    if v is not None:
                        results.append(v)

                if len(results) > 0:
                    total_pred = self.fusion(results)
                    total_loss = self.mse_loss(total_pred, label)
            # -------- 单模态损失部分 --------
            if preds.get("mel") is not None:
                mel_loss = self.mse_loss(preds["mel"] * 15, label.squeeze(1))
            if preds.get("video") is not None:
                video_loss = self.mse_loss(preds["video"] * 15, label.squeeze(1))
            if preds.get("phoneme") is not None:
                phoneme_loss = self.mse_loss(preds["phoneme"] * 15, label.squeeze(1))
            
        elif self.scale_type == "FFMQ": 
            label1 = (label[:,:, :4] - 8).squeeze(1)
            label2 = (label[:,:, 4] - 7)
            if self.num_modalities > 1:
            # 遍历每个模态
                for k, v in preds.items():
                    if v is not None:
                        results.append(v)

                if len(results) > 0:
                    total_pred = self.fusion(results)
                    total_loss = self.mse_loss(total_pred, label)
            # -------- 单模态损失部分 --------
            if preds.get("mel") is not None:
                reg1, reg2 = preds["mel"]
                mel_loss_1 = self.mse_loss(reg1 * 32, label1)
                mel_loss_2 = self.mse_loss(reg2 * 28, label2)
                mel_loss = 0.8 * mel_loss_1 + 0.2 * mel_loss_2
            if preds.get("video") is not None:
                reg1, reg2 = preds["video"]
                video_loss_1 = self.mse_loss(reg1 * 32, label1)
                video_loss_2 = self.mse_loss(reg2 * 28, label2)
                video_loss = 0.8 * video_loss_1 + 0.2 * video_loss_2
            if preds.get("phoneme") is not None:
                reg1, reg2 = preds["phoneme"]
                phoneme_loss_1 = self.mse_loss(reg1 * 32, (label1 - 8).squeeze(1))
                phoneme_loss_2 = self.mse_loss(reg2 * 28, (label2 - 7).squeeze(1))
                phoneme_loss = 0.8 * phoneme_loss_1 + 0.2 * phoneme_loss_2

        elif self.scale_type == "AIS": 
            label1 = (label[:,:, 1:3] - 1).squeeze(1).long()
            label2 = label[:,:,0] - label[:,:,1] - label[:,:,2] - label[:,:,3]
            label2 = torch.concat((label2, label[:,:,3]), dim=-1) - 3
            if self.num_modalities > 1:
            # 遍历每个模态
                for k, v in preds.items():
                    if v is not None:
                        results.append(v)

                if len(results) > 0:
                    total_pred = self.fusion(results)
                    total_loss = self.mse_loss(total_pred, label)
            # -------- 单模态损失部分 --------
            if preds.get("mel") is not None:
                reg, cls_1, cls_2 = preds["mel"]
                mse_loss = self.mse_loss(reg * 9, label2)
                cls_loss_1 = self.ce_loss(cls_1, label1[:,0])
                cls_loss_2 = self.ce_loss(cls_2, label1[:,1])
                mel_loss = mse_loss + cls_loss_1 + cls_loss_2
            if preds.get("video") is not None:
                reg, cls_1, cls_2= preds["video"]
                mse_loss = self.mse_loss(reg * 9, label2)
                cls_loss_1 = self.ce_loss(cls_1, label1[:,0])
                cls_loss_2 = self.ce_loss(cls_2, label1[:,1])
                video_loss = mse_loss + cls_loss_1 + cls_loss_2
            if preds.get("phoneme") is not None:
                reg, cls_1, cls_2= preds["phoneme"]
                mse_loss = self.mse_loss(reg * 9, label2)
                cls_loss_1 = self.ce_loss(cls_1, label1[:,0])
                cls_loss_2 = self.ce_loss(cls_2, label1[:,1])
                phoneme_loss = mse_loss + cls_loss_1 + cls_loss_2
          
        elif self.scale_type == "TFEQ": 
            cls_label = (label[:,:,3] - label[:,:,2] - label[:,:,1] - label[:,:,0] -1).long().squeeze(1)
            reg_label_1 = (label[:,:,0] - 6)
            reg_label_2 = (label[:,:,1] - 8)
            reg_label_3 = (label[:,:,2] - 3)
            if self.num_modalities > 1:
            # 遍历每个模态   
                for k, v in preds.items():
                    if v is not None:
                        results.append(v)

                if len(results) > 0:
                    total_pred = self.fusion(results)
                    total_loss = self.mse_loss(total_pred, label)
            # -------- 单模态损失部分 --------
            if preds.get("mel") is not None:
                reg1, reg2, reg3, cls_1  = preds["mel"]
                mse_loss_1 = self.mse_loss(reg1 * 20, reg_label_1)
                mse_loss_2 = self.mse_loss(reg2 * 24, reg_label_2)
                mse_loss_3 = self.mse_loss(reg3 * 9, reg_label_3)
                cls_loss = self.ce_loss(cls_1, cls_label)
                mel_loss = mse_loss_1 + mse_loss_2 + mse_loss_3 + cls_loss
            if preds.get("video") is not None:
                reg1, reg2, reg3, cls_1  = preds["video"]
                mse_loss_1 = self.mse_loss(reg1 * 20, reg_label_1)
                mse_loss_2 = self.mse_loss(reg2 * 24, reg_label_2)
                mse_loss_3 = self.mse_loss(reg3 * 9, reg_label_3)
                cls_loss = self.ce_loss(cls_1, cls_label)
                video_loss = mse_loss_1 + mse_loss_2 + mse_loss_3 + cls_loss
            if preds.get("phoneme") is not None:
                reg1, reg2, reg3, cls_1  = preds["phoneme"]
                mse_loss_1 = self.mse_loss(reg1 * 20, reg_label_1)
                mse_loss_2 = self.mse_loss(reg2 * 24, reg_label_2)
                mse_loss_3 = self.mse_loss(reg3 * 9, reg_label_3)
                cls_loss = self.ce_loss(cls_1, cls_label)
                phoneme_loss = mse_loss_1 + mse_loss_2 + mse_loss_3 + cls_loss

        elif self.scale_type == "BFI": 
            target_columns = [1,2,3,5,6,7,9,10,11,13,14,15,17,18,19]  # 对应第4、7、10、11列
            # 提取指定列
            label = label[:,:, target_columns]
            label = (label - 4).squeeze(1) 
            if self.num_modalities > 1:
            # 遍历每个模态   
                for k, v in preds.items():
                    if v is not None:
                        results.append(v)

                if len(results) > 0:
                    total_pred = self.fusion(results)
                    total_loss = self.mse_loss(total_pred, label)
            # -------- 单模态损失部分 --------
            if preds.get("mel") is not None:
                reg = preds["mel"]
                mel_loss = self.mse_loss(reg * 16, label)
            
            if preds.get("video") is not None:
                reg = preds["video"]
                video_loss = self.mse_loss(reg * 16, label)
            if preds.get("phoneme") is not None:
                reg1, reg2 = preds["phoneme"]
                phoneme_loss_1 = self.mse_loss(reg1 * 32, (label1 - 8).squeeze(1))
                phoneme_loss_2 = self.mse_loss(reg2 * 28, (label2 - 7).squeeze(1))
                phoneme_loss = 0.8 * phoneme_loss_1 + 0.2 * phoneme_loss_2
        # -------- 融合加权 --------
        # 若 total_loss 不存在，则为 0
    
        fusion_weight = (1 - self.num_modalities * 0.1 ) if total_loss != 0 else 0.0

        loss = fusion_weight * total_loss +  (1 - fusion_weight) * (mel_loss + video_loss + phoneme_loss) / self.num_modalities

        return loss, total_loss, mel_loss, video_loss, phoneme_loss
            

               

