from app.utils.retinaface_detector import RetinaFace_LandmarksDetector, process_video
from app.models.base import BaseModel
from app.utils.video_tower import VitTower
from app.utils.file_util import save_feature_to_mat
# 消除VIT的警告
import logging
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

class Preprocess(BaseModel):
    def __init__(self):
        self.batch_size=None
        self.detector = None
        self.vit_model=None

    def load(self, model_path, batch_size, device):
        self.batch_size = batch_size
        # 声明面部裁剪
        self.detector = RetinaFace_LandmarksDetector()
        # 声明VIT模型
        self.vit_model = VitTower(model_path=model_path, device=device)

    def predict(self, input_path, output_path, mat_path):
        # 提取面部视频
        process_video(
            video_path=input_path, 
            detector=self.detector, 
            batch_size=self.batch_size, 
            output_path=output_path 
            )
        # 提取特征
        features = self.vit_model.forward(output_path, self.batch_size)
        # 将特征保存到mat文件中
        save_feature_to_mat(features=features, mat_path=mat_path)
