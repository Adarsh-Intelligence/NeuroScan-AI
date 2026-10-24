"""Service layer for NeuroBot AI Gemini Integration.

Manages the Google GenAI client, conversation history transformation,
medical safety guardrails, thread offloading, model failover, and error sanitization.
"""
from __future__ import annotations

import asyncio
import time
from typing import List, Optional
from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError, ServerError

from src.chatbot.config import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_TEMPERATURE,
    GEMINI_API_KEY,
    MAX_OUTPUT_TOKENS,
    REQUEST_TIMEOUT_SECONDS,
)
from src.chatbot.prompts import NEUROBOT_SYSTEM_INSTRUCTION
from src.chatbot.schemas import ChatMessage
from src.utils import get_logger

log = get_logger("neurobot.service")


class NeuroBotServiceError(Exception):
    """Base exception for NeuroBot service errors."""
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class NeuroBotService:
    """Manages conversational interactions with Google Gemini API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        # Allow explicit empty string or override
        self.api_key = (GEMINI_API_KEY if api_key is None else api_key).strip()
        self.model_name = (DEFAULT_GEMINI_MODEL if model_name is None else model_name).strip()
        self._client: Optional[genai.Client] = None

        if self.is_configured:
            try:
                self._client = genai.Client(api_key=self.api_key)
                log.info(f"NeuroBotService initialized with model '{self.model_name}'.")
            except Exception as e:
                log.error(f"Failed to initialize GenAI client: {self._mask_key(str(e))}")
                self._client = None
        else:
            log.warning("NeuroBotService initialized without valid GEMINI_API_KEY.")

    @property
    def is_configured(self) -> bool:
        """Check if a non-placeholder API key is configured."""
        if not self.api_key:
            return False
        if self.api_key in ["your_actual_api_key", "your_gemini_api_key_here"]:
            return False
        return True

    @staticmethod
    def _mask_key(text: str) -> str:
        """Helper to ensure raw secrets or keys are never printed in logs or messages."""
        if not text:
            return ""
        import re
        return re.sub(r"AIza[0-9A-Za-z\-_]{35}|AQ\.[0-9A-Za-z\-_]{40,}", "[REDACTED_API_KEY]", text)

    def _build_contents(self, message: str, history: List[ChatMessage]) -> list[types.Content]:
        """Convert validated chat history and new message into Google GenAI Content objects."""
        contents: list[types.Content] = []

        last_role: Optional[str] = None
        for item in history:
            text = (item.content or "").strip()
            if not text:
                continue

            # In Gemini multi-turn format, roles are 'user' or 'model'
            role = "user" if item.role == "user" else "model"

            # Avoid adjacent identical roles if client sent malformed sequence
            if role == last_role:
                continue

            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=text)],
                )
            )
            last_role = role

        # Ensure the final turn is the current user query
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=message)],
            )
        )
        return contents

    def _generate_sync(self, contents: list[types.Content]) -> str:
        """Synchronous API call executed in threadpool with automatic resilience."""
        if not self._client or not self.is_configured:
            raise NeuroBotServiceError(
                message="Gemini API is not configured on the server. Please configure GEMINI_API_KEY.",
                status_code=503,
            )

        config = types.GenerateContentConfig(
            system_instruction=NEUROBOT_SYSTEM_INSTRUCTION,
            temperature=DEFAULT_TEMPERATURE,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )

        # Fallback candidate models if primary model is temporarily overloaded (503 UNAVAILABLE)
        candidate_models = [self.model_name]
        for fallback in ("gemini-3.5-flash-lite", "gemini-3.5-flash"):
            if fallback not in candidate_models:
                candidate_models.append(fallback)

        last_exception: Optional[Exception] = None

        for model_to_try in candidate_models:
            for attempt in range(2):
                try:
                    response = self._client.models.generate_content(
                        model=model_to_try,
                        contents=contents,
                        config=config,
                    )

                    # Extract generated text safely
                    if response and response.text:
                        return response.text.strip()

                    # Handle edge cases (e.g., safety block, empty finish reason)
                    if response and response.candidates:
                        candidate = response.candidates[0]
                        if candidate.finish_reason and candidate.finish_reason.name == "SAFETY":
                            return (
                                "I cannot provide a response to this query because it touches sensitive safety guidelines. "
                                "Please ask an educational question regarding brain anatomy, MRI techniques, or neurological science."
                            )

                    raise NeuroBotServiceError(
                        message="The AI service returned an empty or unparseable response.",
                        status_code=502,
                    )

                except ClientError as ce:
                    last_exception = ce
                    log.warning(f"Gemini ClientError ({ce.code}): {self._mask_key(str(ce))}")
                    if ce.code == 429:
                        raise NeuroBotServiceError(
                            message="The AI service rate limit has been exceeded. Please wait a moment and try again.",
                            status_code=429,
                        )
                    elif ce.code in (401, 403):
                        raise NeuroBotServiceError(
                            message="Authentication with the AI service failed. Please check the server configuration.",
                            status_code=502,
                        )
                    elif ce.code == 404:
                        # Try next fallback model
                        break
                    else:
                        raise NeuroBotServiceError(
                            message="Invalid request sent to the AI service provider.",
                            status_code=400,
                        )

                except ServerError as se:
                    last_exception = se
                    log.warning(f"Gemini ServerError ({se.code}) on model {model_to_try}: {self._mask_key(str(se))}")
                    # If 503 Unavailable / high demand, sleep briefly then try next or retry
                    time.sleep(1.0)
                    continue

                except APIError as ae:
                    last_exception = ae
                    log.error(f"Gemini APIError: {self._mask_key(str(ae))}")
                    raise NeuroBotServiceError(
                        message="Communication error with the AI service provider.",
                        status_code=502,
                    )

                except Exception as e:
                    last_exception = e
                    log.error(f"Unexpected error in NeuroBotService: {self._mask_key(str(e))}", exc_info=True)
                    raise NeuroBotServiceError(
                        message="An unexpected error occurred while processing your request.",
                        status_code=500,
                    )

        # If all candidates exhausted
        if isinstance(last_exception, ServerError):
            raise NeuroBotServiceError(
                message="The AI service is currently experiencing high demand. Please try again shortly.",
                status_code=503,
            )
        raise NeuroBotServiceError(
            message="The AI service provider is currently unavailable. Please try again shortly.",
            status_code=502,
        )

    async def get_reply(self, message: str, history: Optional[List[ChatMessage]] = None) -> str:
        """Asynchronously process user message with conversation context."""
        if not self.is_configured:
            raise NeuroBotServiceError(
                message="Gemini API is not configured on the server. Please set GEMINI_API_KEY in the environment.",
                status_code=503,
            )

        contents = self._build_contents(message, history or [])

        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._generate_sync, contents),
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            log.error(f"Gemini API request timed out after {REQUEST_TIMEOUT_SECONDS}s")
            raise NeuroBotServiceError(
                message="Request to AI service timed out. Please try again.",
                status_code=504,
            )
