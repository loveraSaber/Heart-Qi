import asyncio
import httpx
import websockets
import json

SERVER = "http://127.0.0.1:9097"
WS_SERVER = "ws://127.0.0.1:9097"

async def test():
    # 第一步：提交任务
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{SERVER}/systems/emotion_detect",
            json={
                "input_path": "/path/to/your/video.mp4",
                "output_path": "/path/to/output/"
            }
        )
        data = resp.json()
        print("提交结果:", data)
        task_id = data.get("data",{}).get("task_id")  # 假设返回格式为 {"data": {"task_id": "12345"}, "message": "success", "code": 0}

    if not task_id:
        print("未获取到 task_id，终止")
        return

    # 第二步：WebSocket 监听进度
    print(f"监听任务: {task_id}")
    async with websockets.connect(f"{WS_SERVER}/ws/task/{task_id}") as ws:
        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            print(f"进度: {data}")
            if data.get("progress") == 100:
                print("任务完成:", data)
                break

asyncio.run(test())