from typing import List
from pydantic import BaseModel

class FileRequest(BaseModel):
    input_path: str
    output_path: str
