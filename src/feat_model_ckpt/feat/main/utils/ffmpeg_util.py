import subprocess
from exceptions.handle import ModelException
async def webm2mp4(input_path, output_path):
    try:
        command = [
        "ffmpeg",
        '-hwaccel', 'cuda',
        '-i', input_path,
        '-r', '25',        
        '-c:v', 'h264_nvenc',   # 使用 H.264 视频编码器（MP4 标准）
        '-preset', 'fast',
        '-cq', '8',         # 视频质量（数值越低质量越高）
        '-c:a', 'aac',       # 使用 AAC 音频编码器（MP4 标准）
        '-b:a', '16k',      # 音频比特率
        '-threads', '8',
        '-y', 
        output_path
    ]
        result = subprocess.run(command, check=True)
    except Exception as e:
        raise ModelException(message=f'视频格式转换失败',data=e)
    return True   