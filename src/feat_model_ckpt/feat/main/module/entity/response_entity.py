from typing import List
from pydantic import BaseModel

class ResponseEntity(BaseModel):
    # input_path: str
    angry: float
    disgust: float  
    fear: float
    happiness: float
    sadness: float
    surprise: float
    arousal: float

