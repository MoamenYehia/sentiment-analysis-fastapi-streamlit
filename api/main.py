from fastapi import FastAPI , HTTPException , status

from api.schemas import SentimentRequest , SentimentResponse
from api.client import sentiment_client

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    sentiment_client.initialize_client()
    yield
    sentiment_client.client=None

app=FastAPI(title="Sentiment Analysis API", description="An API for sentiment analysis using Hugging Face models.", version="1.0.0", lifespan=lifespan)

@app.post("/analyze", response_model=SentimentResponse)
def analyze_sentiment(payload:SentimentRequest):
    try:
        result=sentiment_client.analyze_sentiment(payload.text)
        return SentimentResponse(text=payload.text, label=result["label"], score=result["score"])
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    