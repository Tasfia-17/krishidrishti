"""
KrishiDrishti Configuration
Loads settings from environment / .env file.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if present
load_dotenv()


class Settings:
    # llama-server connection (Gemma 4 running via llama.cpp)
    llama_server_host: str = os.getenv("LLAMA_SERVER_HOST", "http://localhost:8080")

    # API server
    host: str = os.getenv("KRISHIDRISHTI_HOST", "0.0.0.0")
    port: int = int(os.getenv("KRISHIDRISHTI_PORT", "7878"))

    # SQLite database
    db_path: Path = Path(os.getenv("DB_PATH", "~/.krishidrishti/diagnoses.db")).expanduser()

    # Upload
    max_upload_size: int = int(os.getenv("MAX_UPLOAD_SIZE", str(10 * 1024 * 1024)))

    # Analysis mode: fast | balanced | accurate
    analysis_mode: str = os.getenv("ANALYSIS_MODE", "balanced")

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    def __post_init__(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
