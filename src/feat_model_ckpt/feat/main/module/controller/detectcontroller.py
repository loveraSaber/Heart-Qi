from fastapi import APIRouter, Depends, Request, Query
from utils.log_util import logger
from utils.response_util import ResponseUtil
from typing import List
from module.entity.request_entity import RequestEntity


featController = APIRouter(prefix='/system')

@featController.post('/detect_video')
async def detect_video(
    request: Request,
    body: RequestEntity
):
    print(f"get request: {body.input_path}")
    result = request.app.state.detector.detect_video(body.input_path, body.output_path)
    logger.info(f"Detection result: {result}")
    return ResponseUtil.success(data=result)

