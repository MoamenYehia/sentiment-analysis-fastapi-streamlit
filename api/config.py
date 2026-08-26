from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    HF_TOKEN: str
    API_port: int

    class Config:
        env_file = ".env"

settings = Settings()