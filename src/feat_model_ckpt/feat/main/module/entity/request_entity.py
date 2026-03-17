from typing import List
from pydantic import BaseModel

class RequestEntity(BaseModel):
    input_path: str
    output_path: str


