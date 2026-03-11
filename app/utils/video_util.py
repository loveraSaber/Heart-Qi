import subprocess
import json
import cv2
import os

# 处理视频
def get_rotate(input_path):
    # ffprobe 命令获取视频旋转信息
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream_tags=rotate",
        "-of", "json",
        input_path
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    except Exception as e:
        print(f"Error occurred while getting video rotation: {e}")
        return 0
    data = json.loads(res.stdout)
    return int(data["streams"][0].get("tags", {}).get("rotate", 0))

def rotate_frame(frame, rotate):
    if rotate == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotate == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotate == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame

def rotate_video(input_path, output_path):
    # 获得视频旋转信息
    rotate = get_rotate(input_path)

    if rotate in [90, 180, 270]:
        # 读取视频并旋转
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_w, out_h = h, w
        # 申明写入对象
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # 旋转每一帧
            frame = rotate_frame(frame, rotate)
            writer.write(frame)
        cap.release()
        writer.release()
        return output_path
    else:
        return input_path
