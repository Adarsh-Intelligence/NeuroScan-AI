"""Configuration for NeuroBot AI Gemini Integration."""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Find and load .env file from project root or current working dir
_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_ROOT / ".env")
load_dotenv()  # also load from cwd if different

# Gemini API Configuration
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "").strip()

# Currently available and fast model on Google GenAI API
# Fallback order can be configured if needed
DEFAULT_GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash").strip()

# Safety & Request Constraints
MAX_MESSAGE_LENGTH: int = int(os.getenv("CHAT_MAX_MESSAGE_LENGTH", "4000"))
MAX_HISTORY_TURNS: int = int(os.getenv("CHAT_MAX_HISTORY_TURNS", "15"))  # last 15 exchanges (30 messages)
MAX_HISTORY_MESSAGE_LENGTH: int = int(os.getenv("CHAT_MAX_HISTORY_MSG_LEN", "2000"))
REQUEST_TIMEOUT_SECONDS: float = float(os.getenv("CHAT_TIMEOUT_SECONDS", "30.0"))
MAX_OUTPUT_TOKENS: int = int(os.getenv("CHAT_MAX_OUTPUT_TOKENS", "1500"))
DEFAULT_TEMPERATURE: float = float(os.getenv("CHAT_TEMPERATURE", "0.3"))
