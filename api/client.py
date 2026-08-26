from huggingface_hub import InferenceClient
from api.config import settings

class HuggingFaceClient:
    def __init__(self):
        self.model_id="cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.client=None
    def initialize_client(self):
        self.client=InferenceClient(api_key=settings.HF_TOKEN)
    def analyze_sentiment(self, text:str) -> dict:
        try :
            response=self.client.text_classification(model=self.model_id, text=text)
            if isinstance(response, list) and len(response) > 0:
                top_result = response[0]
                return {
                    "label": top_result.label,
                    "score": round(float(top_result.score), 4)
                }
            raise ValueError("Unexpected response format from Hugging Face.")
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return {"label": "error", "score": 0.0}

sentiment_client=HuggingFaceClient()             
