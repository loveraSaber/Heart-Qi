import os
from celery import current_task
from app.core.celery_app import celery_app
from app.models.registry import ModelRegistry
import redis

# 从环境变量读取Redis配置，支持Docker容器
redis_host = os.getenv('REDIS_HOST', 'localhost')
redis_port = int(os.getenv('REDIS_PORT', '6379'))
redis_db = int(os.getenv('REDIS_DB', '2'))

redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db, decode_responses=True)

@celery_app.task(bind=True, name='feat_detect')
def feat_detect_task(self, input_path: str, output_path: str, task_id: str):
    """
    异步特征检测任务
    """
    try:
        self.update_state(state='PROCESSING', meta={'progress': 10})
        redis_client.hset(f'task:{task_id}', 'status', 'PROCESSING')
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"文件不存在: {input_path}")
        
        basename = os.path.basename(input_path).split('.')[0]
        output_dir = os.path.join(output_path, basename)
        os.makedirs(output_dir, exist_ok=True)
        
        # 更新进度
        self.update_state(state='PROCESSING', meta={'progress': 50})
        redis_client.hset(f'task:{task_id}', 'progress', 50)
        
        feat_detector = ModelRegistry.get('Feat')
        result = feat_detector.detect_video(
            input_path=input_path,
            output_path=output_dir,
            skip_frames=5,
            num_workers=4,
            batch_size=32
        )
        
        redis_client.hset(f'task:{task_id}', mapping={
            'status': 'SUCCESS',
            'progress': 100,
            'result': str(result)
        })
        
        return {'success': True, 'data': result}
        
    except Exception as e:
        redis_client.hset(f'task:{task_id}', mapping={
            'status': 'FAILED',
            'error': str(e)
        })
        raise
