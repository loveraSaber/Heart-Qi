import os
try:
    from feat import Detector
except ImportError:
    try:
        from feat.detector import Detector
    except ImportError:
        Detector = None
from app.config.env import FeatConfig
from app.config.env import AnalysisConfig
import warnings 
from app.utils.emotion_util import emotion_tendency_pipeline
from app.schemas.response import FeatEntity

warnings.filterwarnings(
    "ignore",
    message="The default value of the antialias parameter"
)

class FeatDetector():
    def __init__(self,):
        if Detector is None:
            self.detector = None
            print("Warning: Detector not available. FeatDetector will not work.")
        else:
            try:
                self.detector = Detector(
                    face_model=FeatConfig.face_model, 
                    landmark_model=FeatConfig.landmark_model, 
                    au_model=FeatConfig.au_model, 
                    emotion_model=FeatConfig.emotion_model, 
                    facepose_model=FeatConfig.facepose_model, 
                    identity_model=FeatConfig.identity_model, 
                    device=FeatConfig.device
                )
            except Exception as e:
                self.detector = None
                print(f"Warning: FeatDetector init failed: {e}")
        self.basic_emotion = FeatConfig.basic_emotion
        self.netral_idx = AnalysisConfig.neutral_idx
        self.baseline = AnalysisConfig.baseline
        self.lambda_scale = AnalysisConfig.lambda_scale
        self.temperature = AnalysisConfig.temperature


    def detect_video(self, input_path, output_path, skip_frames=5, num_workers=4, batch_size=64):
        input_path.endswith('.mp4')
        # 检测视频文件是否存在
        if not os.path.exists(input_path):
            print(f"Video file {input_path} does not exist.")
            return False
        if not input_path.endswith('.mp4'):
            print(f"File {input_path} is not an mp4 video.")
            return False
        # input_path = '/vdb/structure/test_data/input/20260121/0001-4548484135.mp4'    
        base_path = os.path.splitext(os.path.basename(input_path))[0]
        # 如果文件不存在，则新建目录
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        prediction = self.detector.detect_video(
            input_path, 
            num_workers=num_workers, 
            skip_frames=skip_frames, 
            batch_size=batch_size,
            pin_memory=True
            )
        # Save raw prediction results to CSV
        prediction.to_csv(
            os.path.join(output_path, f"{base_path}.csv"),
            index=False,        # 不保存行索引
            encoding="utf-8-sig"  # Windows / Excel 友好 
        )
        # 删除包含空值的行
        prediction = prediction.dropna(subset=self.basic_emotion)
        # 提取 logits 进行情感倾向分析
        logits = prediction[self.basic_emotion].values

        emotion, arousal = emotion_tendency_pipeline(
            logits=logits, 
            neutral_idx=self.netral_idx, 
            baseline=self.baseline, 
            lambda_scale=self.lambda_scale, 
            temperature=self.temperature
            )
        
        result = FeatEntity(
            angry=emotion[0],
            disgust=emotion[1],
            fear=emotion[2],
            happiness=emotion[3],
            sadness=emotion[4],
            surprise=emotion[5],
            arousal=arousal
        )
        
        return result
