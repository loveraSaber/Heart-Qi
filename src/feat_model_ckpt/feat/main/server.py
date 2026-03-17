from contextlib import asynccontextmanager
from fastapi import FastAPI
from config.env import AppConfig
from module.controller.detectcontroller import featController
from utils.log_util import logger
from pyfeat.detector import EmotionDetector

# 生命周期事件
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f'{AppConfig.app_name}开始启动')
    logger.info('启动Detector')
    app.state.detector = EmotionDetector()
    logger.info(f'{AppConfig.app_name}启动成功')
    yield
    logger.info(f'{AppConfig.app_name}正在关闭')
   
# 初始化FastAPI对象
app = FastAPI(
    title=AppConfig.app_name,
    description=f'{AppConfig.app_name}接口文档',
    version=AppConfig.app_version,
    lifespan=lifespan,
)

# 加载路由列表
controller_list = [
    {'router': featController, 'tags': ['情感检测模块']}
]

for controller in controller_list:
    app.include_router(router=controller.get('router'), tags=controller.get('tags'))
