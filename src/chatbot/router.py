"""FastAPI Router for NeuroBot AI Endpoints."""
from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, Body, HTTPException, Request, status
from fastapi.responses import JSONResponse

from src.chatbot.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatStatusResponse,
)
from src.chatbot.service import NeuroBotService, NeuroBotServiceError
from src.utils import get_logger

log = get_logger("neurobot.router")

router = APIRouter(prefix="/api/chat", tags=["NeuroBot AI"])

# Global service instance (lazily initialized or reloaded)
_service: Optional[NeuroBotService] = None


def get_chat_service() -> NeuroBotService:
    global _service
    if _service is None:
        _service = NeuroBotService()
    return _service


def set_chat_service(service: NeuroBotService) -> None:
    """Useful for dependency injection or testing."""
    global _service
    _service = service


@router.get(
    "/status",
    response_model=ChatStatusResponse,
    summary="Check NeuroBot AI service readiness and configuration",
)
async def get_chatbot_status():
    """Returns the operational status of the Gemini chatbot service."""
    service = get_chat_service()
    is_ready = service.is_configured
    return ChatStatusResponse(
        status="operational" if is_ready else "unconfigured",
        service="NeuroBot AI",
        provider="Google Gemini",
        model=service.model_name,
        configured=is_ready,
        version="1.0.0",
    )


@router.post(
    "",
    response_model=ChatResponse,
    summary="Interact with NeuroBot AI medical education chatbot",
    responses={
        200: {
            "description": "Educational reply generated successfully.",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "reply": "Glioma is a type of brain tumor that originates in the glial cells.",
                        "model": "gemini-3.5-flash",
                        "timestamp": "2026-10-24T07:30:00Z",
                    }
                }
            },
        },
        400: {"description": "Invalid request or empty message."},
        422: {"description": "Unprocessable entity."},
        429: {"description": "AI provider rate limit exceeded."},
        502: {"description": "Upstream AI provider error."},
        503: {"description": "AI service is unconfigured or unavailable."},
        504: {"description": "AI service request timed out."},
    },
)
async def chat_with_neurobot(payload: ChatRequest = Body(...)) -> ChatResponse:
    """Processes user queries about neuroscience, brain tumors, MRI terms, and AI classification.

    Accepts multi-turn conversation history and generates educational responses using Google Gemini API.
    """
    # Defensive validation: ensure message contains meaningful text
    cleaned_message = payload.message.strip() if payload.message else ""
    if not cleaned_message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message cannot be empty or contain only whitespace.",
        )

    service = get_chat_service()

    try:
        reply_text = await service.get_reply(
            message=cleaned_message,
            history=payload.history,
        )
        return ChatResponse(
            success=True,
            reply=reply_text,
            model=service.model_name,
        )
    except NeuroBotServiceError as nse:
        raise HTTPException(
            status_code=nse.status_code,
            detail=nse.message,
        )
    except Exception as e:
        log.error(f"Unhandled error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal server error occurred while processing the chat request.",
        )
