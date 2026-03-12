import argparse
import os
import sys
from dotenv import load_dotenv
from functools import lru_cache
from pydantic_settings import BaseSettings
from typing import Literal


class AppSettings(BaseSettings):
    """
    应用配置
    """

    app_env: str = 'dev'
    app_name: str = 'Dash-FasAPI-Admin'
    app_root_path: str = ''
    app_host: str = '0.0.0.0'
    app_port: int = 9096
    app_version: str = '2.0.0'
    app_reload: bool = True
    app_ip_location_query: bool = True
    app_same_time_login: bool = True


class ModelSettings(BaseSettings):
    """
    模型配置
    """
    vit_base_patch: str = '/app/app/src/vit-base-patch16-224'
    BPS_Model_path: str = '/app/app/src/model_ckpt/BPS/last.ckpt'
    CPSS_Model_path: str = '/app/app/src/model_ckpt/CPSS/last-v2.ckpt'
    PHQ_Model_path: str = '/app/app/src/model_ckpt/PHQ/last.ckpt'
    STAI_Model_path: str = '/app/app/src/model_ckpt/STAI/last.ckpt'
    BPS_Config_path: str = '/app/app/core/configs/model_config/BPS.yaml'
    CPSS_Config_path: str = '/app/app/core/configs/model_config/CPSS.yaml'
    PHQ_Config_path: str = '/app/app/core/configs/model_config/PHQ.yaml'
    STAI_Config_path: str = '/app/app/core/configs/model_config/STAI.yaml'
    
    model_device: Literal['cpu', 'cuda'] = 'cuda'
    batch_size: int = 32

class PyFeatSettings(BaseSettings):
    """
    Py-Feat 配置
    """

    face_model: str = 'img2pose'
    landmark_model: str = 'mobilefacenet'
    au_model: str = 'xgb'
    emotion_model: str = 'resmasknet'
    facepose_model: str = 'img2pose'
    identity_model: str = 'facenet'
    device: str = 'cuda'  # 'cpu' or 'cuda'
    basic_emotion: list = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']

class AnalysisSettings(BaseSettings):
    """
    分析配置
    """
    neutral_idx: int = 6
    baseline: float = 0.05
    lambda_scale: float = 3.0
    temperature: float = 2.0


class VideoProcessSetting(BaseSettings):
    crop_height: int = 200
    crop_width: int = 200
    start_id: int = 1
    stop_id: int = 5


class RedisSettings(BaseSettings):
    """
    Redis配置
    """

    redis_host: str = '127.0.0.1'
    redis_port: int = 6379
    redis_username: str = ''
    redis_password: str = ''
    redis_db_broker:int=0
    redis_db_backend:int=1
    redis_db:int=2


class UploadSettings:
    """
    上传配置
    """

    UPLOAD_PREFIX = '/profile'
    UPLOAD_PATH = 'df_admin/upload_path'
    UPLOAD_MACHINE = 'A'
    DEFAULT_ALLOWED_EXTENSION = [
        # 图片
        'bmp',
        'gif',
        'jpg',
        'jpeg',
        'png',
        # word excel powerpoint
        'doc',
        'docx',
        'xls',
        'xlsx',
        'ppt',
        'pptx',
        'html',
        'htm',
        'txt',
        # 压缩文件
        'rar',
        'zip',
        'gz',
        'bz2',
        # 视频格式
        'mp4',
        'avi',
        'rmvb',
        # pdf
        'pdf',
    ]
    DOWNLOAD_PATH = 'df_admin/download_path'

    def __init__(self):
        if not os.path.exists(self.UPLOAD_PATH):
            os.makedirs(self.UPLOAD_PATH)
        if not os.path.exists(self.DOWNLOAD_PATH):
            os.makedirs(self.DOWNLOAD_PATH)


class CachePathConfig:
    """
    缓存目录配置
    """

    PATH = os.path.join(os.path.abspath(os.getcwd()), 'caches')
    PATHSTR = 'caches'


class GetConfig:
    """
    获取配置
    """

    def __init__(self):
        self.parse_cli_args()

    @lru_cache()
    def get_app_config(self):
        """
        获取应用配置
        """
        # 实例化应用配置模型
        return AppSettings()
    
    @lru_cache()
    def get_redis_config(self):
        return RedisSettings()
    
    @lru_cache()
    def get_pyfeat_config(self):
        """
        获取Py-Feat配置
        """
        return PyFeatSettings()
    
    @lru_cache()
    def get_analysis_config(self):
        """
        获取分析配置
        """
        return AnalysisSettings()

    @lru_cache()
    def get_model_config(self):
        """
        获取模型配置
        """
        # 实例化模型配置模型
        return ModelSettings()

    @lru_cache()
    def get_video_processer_conifg(self):
        """
        获取视频裁剪参数
        """
        return VideoProcessSetting()

    @staticmethod
    def parse_cli_args():
        """
        解析命令行参数
        """
        if 'uvicorn' in sys.argv[0] or 'celery' in sys.argv[0]:
            # uvicorn 和 celery 启动时都跳过 argparse
            pass
        else:
            # 使用argparse定义命令行参数
            parser = argparse.ArgumentParser(description='命令行参数')
            parser.add_argument('--env', type=str, default='', help='运行环境')
            # 解析命令行参数
            args = parser.parse_args()
            # 设置环境变量，如果未设置命令行参数，默认APP_ENV为dev
            os.environ['APP_ENV'] = args.env if args.env else 'dev'
        # 读取运行环境
        run_env = os.environ.get('APP_ENV', '')
        # 运行环境未指定时默认加载.env.dev
        env_file = '.env.dev'
        # 运行环境不为空时按命令行参数加载对应.env文件
        if run_env != '':
            env_file = f'.env.{run_env}'
        # 加载配置
        load_dotenv(env_file)


# 实例化获取配置类
get_config = GetConfig()
# 应用配置
AppConfig = get_config.get_app_config()
# 模型配置
ModelConfig = get_config.get_model_config()
# 视频裁剪配置
VideoConfig = get_config.get_video_processer_conifg()

RedisConfig = get_config.get_redis_config()

FeatConfig = get_config.get_pyfeat_config()

AnalysisConfig = get_config.get_analysis_config()
