from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    project_root: Path = Path(__file__).resolve().parents[1]
    #: Primary training data (challenge dataset)
    claim_data_xlsx: Path = project_root / "claim_use_case_dataset.xlsx"
    artifacts_dir: Path = project_root / "artifacts"

    #: Google AI Studio — ``generateContent`` for Gemini, Gemma (hosted), etc.
    gemini_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_AI_API_KEY", "GOOGLE_API_KEY"),
    )
    gemini_model: str = Field(
        default="gemma-4-31b-it",
        validation_alias=AliasChoices("GEMINI_MODEL", "GOOGLE_GEMINI_MODEL"),
    )
    google_cloud_project: str | None = Field(
        default=None,
        validation_alias=AliasChoices("GOOGLE_CLOUD_PROJECT", "GCP_PROJECT"),
    )
    #: Cap hosted generateContent rate (spacing = 60 / rpm between completed calls).
    gemini_rpm: float | None = Field(default=None, validation_alias=AliasChoices("GEMINI_RPM"))
    #: Override spacing between generateContent calls (seconds). Takes precedence over ``gemini_rpm`` when set.
    gemini_min_request_interval_s: float | None = Field(
        default=None,
        validation_alias=AliasChoices("GEMINI_MIN_REQUEST_INTERVAL_S"),
    )

    #: Optional OpenAI-compatible fallback (not used when Gemini key is set)
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_chat_model: str = "gpt-4o-mini"

    log_level: str = "INFO"

    def gemini_request_min_interval_s(self) -> float | None:
        if self.gemini_min_request_interval_s is not None and self.gemini_min_request_interval_s > 0:
            return float(self.gemini_min_request_interval_s)
        if self.gemini_rpm is not None and self.gemini_rpm > 0:
            return 60.0 / float(self.gemini_rpm)
        return None


@lru_cache
def get_settings() -> Settings:
    return Settings()
