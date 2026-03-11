from fastapi import APIRouter, Request
from app.schemas.filerequest import FileRequest
from app.utils.response_util import ResponseUtil
from app.tasks.emotion_task import emotion_detect_task
from app.core.celery_app import celery_app
import uuid
import redis
import os

emotionController = APIRouter(prefix='/systems')

# 从环境变量读取Redis配置，支持Docker容器
redis_host = os.getenv('REDIS_HOST', 'localhost')
redis_port = int(os.getenv('REDIS_PORT', '6379'))
redis_db = int(os.getenv('REDIS_DB', '2'))

redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db, decode_responses=True)

@emotionController.post("/emotion_detect")
async def detect_emotion(request: Request, body: FileRequest):
    """
    提交情感检测任务（异步）
    返回任务ID，客户端通过 WebSocket 接收结果
    """
    task_id = str(uuid.uuid4())
    
    # 提交 Celery 任务
    celery_task = emotion_detect_task.delay(
        input_path=body.input_path,
        output_path=body.output_path,
        task_id=task_id
    )
    
    # 存储任务映射关系
    redis_client.set(f'celery_task:{task_id}', celery_task.id, ex=86400)
    
    return ResponseUtil.success(data={
        'task_id': task_id,
        'status': 'QUEUED',
        'message': '任务已提交，请通过 WebSocket 连接获取结果'
    })
