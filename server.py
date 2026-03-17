# from contextlib import asynccontextmanager

# from contextlib import asynccontextmanager
# ======== Numpy 2.0 向后兼容补丁已移除 ========
# ========================================

from fastapi import FastAPI
from app.config.env import AppConfig
from app.api.emotion_controller import emotionController
from app.api.feat_controller import featController
from app.api.websocket_handler import websocketRouter
from app.models.model_factory import ModelFactory
from app.models.registry import ModelRegistry
'''现在模型通过 Celery worker 的 worker_process_init 信号加载，不需要在这里加载了，浪费GPU资源'''
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     print('System loading...')
#     preprocess = ModelFactory.create_model('Preprocess')
#     cpss_model = ModelFactory.create_model('CPSS')
#     phq_model = ModelFactory.create_model('PHQ')
#     stai_model = ModelFactory.create_model('STAI')
#     feat_detector = ModelFactory.create_model('Feat')
    
#     ModelRegistry.register('preprocess', preprocess)
#     ModelRegistry.register("CPSS", cpss_model)
#     ModelRegistry.register("PHQ", phq_model)
#     ModelRegistry.register("STAI", stai_model)
#     ModelRegistry.register("Feat", feat_detector)

#     yield    
#     print("System shutdown...")
'''声明API应用实例'''
app = FastAPI(
    title=AppConfig.app_name,
    description=f'{AppConfig.app_name}接口文档',
    version=AppConfig.app_version,
    #lifespan=lifespan,
)

# 注册 HTTP 路由
controller_list = [
    {'router': emotionController, 'tags': ['情感处理模块']},
    {'router': featController, 'tags': ['特征检测模块']},
]

for controller in controller_list:
    app.include_router(router=controller.get('router'), tags=controller.get('tags'))

# 注册 WebSocket 路由
app.include_router(websocketRouter)