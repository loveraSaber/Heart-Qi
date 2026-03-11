import torch
import numpy as np

class LabelCalculate:
    def __init__(self, config):
        self.scale_type = config["train_scale_type"]

    def calculate(self, label_score, label_num):

        B, label_dim = label_score.shape
        label_score = label_score.detach().cpu().numpy()

        label_batch = []
        label_num_batch = []
        label_score_batch = []

        if self.scale_type == "AIS":
            for i in range(B):
                scores = label_score[i]

                # ---- 计算各项分数 ----
                score_1 = float(np.sum(scores)) 
                score_2 = float(scores[1])       
                score_3 = float(scores[2])      
                score_4 = float(scores[3])       

                label_score_t = [score_1, score_2, score_3, score_4]
                label_score_batch.append(label_score_t)

                label_list = []
                label_num_list = []

                # 1
                if score_1 < 12:
                    label_list.append("无睡眠障碍")
                    label_num_list.append(1)
                elif score_1 < 14:
                    label_list.append("存在失眠障碍风险")
                    label_num_list.append(2)
                else:
                    label_list.append("存在睡眠障碍")
                    label_num_list.append(3)

                # 2
                if score_2 <= 1:
                    label_list.append("睡眠时间充足")
                    label_num_list.append(4)
                elif score_2 < 3:
                    label_list.append("睡眠时间不足")
                    label_num_list.append(5)
                else:
                    label_list.append("睡眠时间严重不足")
                    label_num_list.append(6)

                # 3
                if score_3 <= 1:
                    label_list.append("睡眠质量良好")
                    label_num_list.append(7)
                elif score_3 < 3:
                    label_list.append("睡眠质量一般或较差")
                    label_num_list.append(8)
                else:
                    label_list.append("睡眠质量差")
                    label_num_list.append(9)

                # 4
                if score_4 <= 3:
                    label_list.append("日间功能良好")
                    label_num_list.append(10)
                elif score_4 < 6:
                    label_list.append("存在日间功能障碍风险")
                    label_num_list.append(11)
                else:
                    label_list.append("存在日间功能障碍")
                    label_num_list.append(12)

                label_batch.append(label_list)
                label_num_batch.append(label_num_list)

        else:
            raise ValueError(f"未知的量表类型: {self.scale_type}")

        return label_batch, label_num_batch, label_score_batch
