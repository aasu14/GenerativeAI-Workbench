import os
from pydantic_settings import BaseSettings
from pydantic import field_validator
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    openai_api_key: str = ""
    telegram_bot_token: str = ""
    database_url: str = "sqlite+aiosqlite:///./agents.db"
    openai_model: str = "gpt-4o-mini"
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    model_config = {"env_file": ".env"}

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
