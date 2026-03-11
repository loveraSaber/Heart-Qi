import torch
from torch import nn

class SimpleClassifier(nn.Module):
    """简单的5分类模型"""
    
    def __init__(self, input_dim=768, hidden_dims=[512, 256, 128], num_classes=5, dropout_rate=0.3):
        super(SimpleClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 构建隐藏层
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(prev_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, T, 768) 或 (batch_size, 768)
        if x.dim() == 3:
            # 如果是时间序列数据，取最后一个时间步或者使用全局平均池化
            x = x[:, -1, :]  # 取最后一个时间步
            # 或者: x = x.mean(dim=1)  # 全局平均池化
            
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output