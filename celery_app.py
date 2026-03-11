from celery import Celery
from config import app_config
celery_app=Celery(
    'visualmodel',
    broker=f'',
    backend=f'',
    include=['','']
)
celery_app.conf.update(task_serializer='json',
                       accept_content=['json'],
                       result_serializer='json',
                       timezone='Asia/Shanghai',
                       enable_utc=True,
                       task_track_started=True,
                       taske_time_limit=3600,)