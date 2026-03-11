from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from app.utils.response_util import ResponseUtil
import redis
import json
import asyncio
import os

websocketRouter = APIRouter()

# 从环境变量读取Redis配置，支持Docker容器
redis_host = os.getenv('REDIS_HOST', 'localhost')
redis_port = int(os.getenv('REDIS_PORT', '6379'))
redis_db = int(os.getenv('REDIS_DB', '2'))

redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db, decode_responses=True)

@websocketRouter.websocket("/ws/task/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    """
    WebSocket 端点：实时推送任务进度和结果
    """
    await websocket.accept()
    
    try:
        while True:
            # 从 Redis 获取任务状态
            task_data = redis_client.hgetall(f'task:{task_id}')
            
            if not task_data:
                await websocket.send_json({
                    'code': 404,
                    'msg': '任务不存在',
                    'status': 'NOTFOUND'
                })
                break
            
            status = task_data.get('status', 'UNKNOWN')
            progress = int(task_data.get('progress', 0))
            
            # 发送进度状态
            if status == 'PROCESSING':
                await websocket.send_json({
                    'code': 0,
                    'msg': '处理中...',
                    'status': 'PROCESSING',
                    'progress': progress
                })
            
            # 任务完成
            elif status == 'SUCCESS':
                result_str = task_data.get('result', '{}')
                result = eval(result_str) if result_str != '{}' else {}
                
                await websocket.send_json({
                    'code': 0,
                    'msg': '处理完成',
                    'status': 'SUCCESS',
                    'progress': 100,
                    'data': result
                })
                
                # 清理 Redis 数据
                redis_client.delete(f'task:{task_id}')
                break
            
            # 任务失败
            elif status == 'FAILED':
                error = task_data.get('error', 'Unknown error')
                await websocket.send_json({
                    'code': -1,
                    'msg': f'处理失败: {error}',
                    'status': 'FAILED'
                })
                redis_client.delete(f'task:{task_id}')
                break
            
            # 等待下次查询
            await asyncio.sleep(2)
    
    except WebSocketDisconnect:
        print(f"Client disconnected from task {task_id}")
    except Exception as e:
        await websocket.send_json({
            'code': -1,
            'msg': f'错误: {str(e)}'
        })
