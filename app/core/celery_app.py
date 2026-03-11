import os
from celery import Celery
from app.config.env import AppConfig

# 从环境变量读取Redis配置，支持Docker容器
redis_host = os.getenv('REDIS_HOST', 'localhost')
redis_port = os.getenv('REDIS_PORT', '6379')
redis_db_broker = os.getenv('REDIS_DB_BROKER', '0')      # 任务队列数据库
redis_db_backend = os.getenv('REDIS_DB_BACKEND', '1')    # 任务结果数据库

broker_url = f'redis://{redis_host}:{redis_port}/{redis_db_broker}'
backend_url = f'redis://{redis_host}:{redis_port}/{redis_db_backend}'

celery_app = Celery(
    'visualmodel',
    broker=broker_url,
    backend=backend_url,
    include=['app.tasks.emotion_task', 'app.tasks.feat_task']
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Asia/Shanghai',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,
)
