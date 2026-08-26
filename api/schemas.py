from pydantic import BaseModel , Field

class SentimentRequest(BaseModel):
    text: str=Field(..., description="The text to analyze for sentiment.")

class SentimentResponse(BaseModel):
    text:str
    label:str
    score:float    