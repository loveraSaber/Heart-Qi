from typing import List
from pydantic import BaseModel

class EmotionEntity(BaseModel):
    pressure: float
    state_anxiety: float
    trait_anxiety: float
    depression: float

class FeatEntity(BaseModel):
    # input_path: str
    angry: float
    disgust: float  
    fear: float
    happiness: float
    sadness: float
    surprise: float
    arousal: float
