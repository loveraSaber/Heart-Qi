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

@celery_app.task(bind=True, name='emotion_task')
def emotion_detect_task(self, input_path: str, output_path: str, task_id: str):
    """
    异步情感检测任务
    """
    try:
        # 更新任务状态为处理中
        self.update_state(state='PROCESSING', meta={'progress': 10})
        redis_client.hset(f'task:{task_id}', 'status', 'PROCESSING')
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"文件不存在: {input_path}")
        
        basename = os.path.basename(input_path).split('.')[0]
        output_dir = os.path.join(output_path, basename)
        os.makedirs(output_dir, exist_ok=True)
        
        mp4_output = os.path.join(output_dir, f'{basename}.mp4')
        mat_path = os.path.join(output_dir, f'{basename}.mat')
        
        # 30% 进度：预处理
        self.update_state(state='PROCESSING', meta={'progress': 30})
        redis_client.hset(f'task:{task_id}', 'progress', 30)
        processor = ModelRegistry.get('preprocess')
        processor.predict(input_path=input_path, output_path=mp4_output, mat_path=mat_path)
        
        # 60% 进度：模型推断
        self.update_state(state='PROCESSING', meta={'progress': 60})
        redis_client.hset(f'task:{task_id}', 'progress', 60)
        cpss = ModelRegistry.get('CPSS')
        phq = ModelRegistry.get('PHQ')
        stai = ModelRegistry.get('STAI')
        
        pressure = cpss.predict(mat_path)
        state_anxiety, trait_anxiety = stai.predict(mat_path)
        depression = phq.predict(mat_path)
        
        # 100% 进度：完成
        result = {
            'pressure': round(pressure.item(), 2),
            'state_anxiety': round(state_anxiety.item(), 2),
            'trait_anxiety': round(trait_anxiety.item(), 2),
            'depression': round(depression.item(), 2),
        }
        
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
