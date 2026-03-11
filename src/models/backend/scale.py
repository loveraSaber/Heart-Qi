import torch.nn as nn

class STAI_Prediction(nn.Module):
    def __init__(self, input_dim):
        super(STAI_Prediction, self).__init__()

        self.avgpooling = nn.AdaptiveAvgPool1d(1)
        self.reg_head = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 2),
            nn.Sigmoid()
        )
        # #分类共享特征层
        # self.cls_head = nn.Sequential(
        #     nn.Linear(input_dim, 32),
        #     nn.ReLU(),
        #     nn.Dropout(0.1)
        # )
        # self.cls_head_1 = nn.Linear(32,3)
        # self.cls_head_2 = nn.Linear(32,3)
    def forward(self, data):
        #下采样时间维度
        data = self.avgpooling(data.transpose(1,2)).squeeze(-1)
        #回归任务预测
        reg_pred = self.reg_head(data)
        #获得共享特征
        # cls_feat = self.cls_head(data)
        #分类任务预测
        # cls_pred_1 = self.cls_head_1(cls_feat)
        # cls_pred_2 = self.cls_head_2(cls_feat)
        return reg_pred
    


class PHQ_Prediction(nn.Module):
    def __init__(self, input_dim):
        super(PHQ_Prediction, self).__init__()

        self.avgpooling = nn.AdaptiveAvgPool1d(1)
        self.reg_head = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        # #分类共享特征层
        # self.cls_head = nn.Sequential(
        #     nn.Linear(input_dim, 32),
        #     nn.ReLU(),
        #     nn.Dropout(0.1)
        # )
        # self.cls_head_1 = nn.Linear(32,3)
        # self.cls_head_2 = nn.Linear(32,3)
    def forward(self, data):
        #下采样时间维度
        data = self.avgpooling(data.transpose(1,2)).squeeze(-1)
        #回归任务预测
        reg_pred = self.reg_head(data)
        
        return reg_pred
    
class CPSS_Prediction(nn.Module):
    def __init__(self, input_dim):
        super(CPSS_Prediction, self).__init__()

        self.avgpooling = nn.AdaptiveAvgPool1d(1)
        self.reg_head = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        # #分类共享特征层
        # self.cls_head = nn.Sequential(
        #     nn.Linear(input_dim, 32),
        #     nn.ReLU(),
        #     nn.Dropout(0.1)
        # )
        # self.cls_head_1 = nn.Linear(32,3)
        # self.cls_head_2 = nn.Linear(32,3)
    def forward(self, data):
        #下采样时间维度
        data = self.avgpooling(data.transpose(1,2)).squeeze(-1)
        #回归任务预测
        reg_pred = self.reg_head(data)
        
        return reg_pred
    
class BPS_Prediction(nn.Module):
    def __init__(self, input_dim):
        super(BPS_Prediction, self).__init__()

        self.avgpooling = nn.AdaptiveAvgPool1d(1)
        self.reg_head = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 4),
            nn.Sigmoid()
        )
   
    def forward(self, data):
        #下采样时间维度
        data = self.avgpooling(data.transpose(1,2)).squeeze(-1)
        #回归任务预测
        reg_pred = self.reg_head(data)
        
        return reg_pred
    
class FFMQ_Prediction(nn.Module):
    def __init__(self, input_dim):
        super(FFMQ_Prediction, self).__init__()

        self.avgpooling = nn.AdaptiveAvgPool1d(1)
        #共享特征层
        self.reg_head = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        #回归预测层
        self.reg_head1 = nn.Sequential(
            nn.Linear(32, 4),
            nn.Sigmoid()
        )
        self.reg_head2 = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        # #分类共享特征层
        # self.cls_head = nn.Sequential(
        #     nn.Linear(input_dim, 32),
        #     nn.ReLU(),
        #     nn.Dropout(0.1)
        # )
        # self.cls_head_1 = nn.Linear(32,3)
        # self.cls_head_2 = nn.Linear(32,3)
    def forward(self, data):
        #下采样时间维度
        data = self.avgpooling(data.transpose(1,2)).squeeze(-1)
        #回归任务预测
        reg_pred = self.reg_head(data)
        reg1 = self.reg_head1(reg_pred)
        reg2 = self.reg_head2(reg_pred)
        
        return (reg1, reg2)
    

class AIS_Prediction(nn.Module):
    def __init__(self, input_dim):
        super(AIS_Prediction, self).__init__()

        self.avgpooling = nn.AdaptiveAvgPool1d(1)

        #共享特征层
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        #回归头
        self.reg_head = nn.Sequential(
            nn.Linear(16, 2),
            nn.Sigmoid()
        )
        #分类头
        self.cls_head_1 = nn.Linear(16,4)
        self.cls_head_2 = nn.Linear(16,4)

    def forward(self, data):
        #下采样时间维度
        data = self.avgpooling(data.transpose(1,2)).squeeze(-1)
        #回归任务预测
        feature = self.projection(data)
        reg_head = self.reg_head(feature)
        cls_1 = self.cls_head_1(feature)
        cls_2 = self.cls_head_2(feature)
    
        return (reg_head, cls_1, cls_2)

class TFEQ_Prediction(nn.Module):
    def __init__(self, input_dim):
        super(TFEQ_Prediction, self).__init__()

        self.avgpooling = nn.AdaptiveAvgPool1d(1)

        #共享特征层
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        #回归头
        self.reg_head_1 = nn.Sequential(
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        self.reg_head_2 = nn.Sequential(
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        self.reg_head_3 = nn.Sequential(
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        #分类头
        self.cls_head = nn.Linear(16,4)
    

    def forward(self, data):
        #下采样时间维度
        data = self.avgpooling(data.transpose(1,2)).squeeze(-1)
        #回归任务预测
        feature = self.projection(data)
        reg_1 = self.reg_head_1(feature)
        reg_2 = self.reg_head_2(feature)
        reg_3 = self.reg_head_3(feature)
        cls_1 = self.cls_head(feature)
        return (reg_1, reg_2, reg_3, cls_1)


class BFI_Prediction(nn.Module):
    def __init__(self, input_dim):
        super(BFI_Prediction, self).__init__()

        self.avgpooling = nn.AdaptiveAvgPool1d(1)

        #共享特征层
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 15),
            nn.Sigmoid()
        )
    

    def forward(self, data):
        #下采样时间维度
        data = self.avgpooling(data.transpose(1,2)).squeeze(-1)
        #回归任务预测
        reg = self.projection(data)   
        return reg