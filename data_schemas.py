from pydantic import BaseModel
from typing import List, Tuple, TypedDict

# Required for Gemini's structured output
class AttentionElement(BaseModel):
    label: str
    attention_score: float

# Required for Gemini's structured output
class AttentionAnalysis(BaseModel):
    elements: List[AttentionElement]

class AOIS(TypedDict):
    label: str
    attention_score: float
    bounding_box: Tuple[int, int, int, int]
    
class BoundingBox(TypedDict):
    box_2d: Tuple[int, int, int, int] # !!!Gemini returns: y1, x1, y2, x2
    label: str
    