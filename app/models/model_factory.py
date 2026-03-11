from app.models.bps import BPSModel
from app.models.cpss import CPSSModel
from app.models.phq import PHQModel
from app.models.preprocess import Preprocess
from app.config.env import ModelConfig
from app.models.stai import STAIModel
from app.models.feat_detector import FeatDetector

class ModelFactory:
    def create_model(model_type):
        if model_type == 'CPSS':
            cpss = CPSSModel()
            cpss.load(ModelConfig.CPSS_Config_path, ModelConfig.CPSS_Model_path, ModelConfig.model_device)
            return cpss
        elif model_type == 'BPS':
            bpsmodel = BPSModel()
            bpsmodel.load(ModelConfig.BPS_Config_path, ModelConfig.BPS_Model_path, ModelConfig.model_device)
            return bpsmodel
        elif model_type == 'PHQ':
            phqmodel = PHQModel()
            phqmodel.load(ModelConfig.PHQ_Config_path, ModelConfig.PHQ_Model_path, ModelConfig.model_device)
            return phqmodel
        elif model_type == 'STAI':
            staimodel = STAIModel()
            staimodel.load(ModelConfig.STAI_Config_path, ModelConfig.STAI_Model_path, ModelConfig.model_device)
            return staimodel 
        elif model_type == 'Preprocess':
            preprocess = Preprocess()
            preprocess.load(ModelConfig.vit_base_patch, ModelConfig.batch_size, ModelConfig.model_device)
            return preprocess
        elif model_type == 'Feat':
            feat_detector = FeatDetector()
            return feat_detector
        else:
            print('输入模型不正确，请检查参数类型')
