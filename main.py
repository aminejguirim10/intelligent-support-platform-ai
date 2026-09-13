import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from file_parser import extract_tickets_or_text
from ai_agent import run_analysis, split_tickets
from models import AnalysisResponse, AnalyzedTicket
from agent_models import AgentChatRequest, AgentChatResponse
from support_agent import run_support_agent_chat
from backend_client import BackendClient

logger = logging.getLogger("ticket-ai")

app = FastAPI(title="Ticket AI Analysis Service")

# Configure CORS
cors_origins = os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:4200").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool for CPU/IO bound LLM calls — shared across all requests
_executor = ThreadPoolExecutor(max_workers=8)

# Concurrency limiter — never more than 5 LLM analysis calls in flight at once
_ANALYSIS_SEMAPHORE = asyncio.Semaphore(5)

# Batch size for concurrent analysis
_BATCH_SIZE = 5


class TextAnalysisRequest(BaseModel):
    text: str


# ---------------------------------------------------------------------------
# Batched analysis helper — processes tickets in controlled batches
# ---------------------------------------------------------------------------

async def _analyze_single(ticket: str) -> AnalysisResponse:
    """Analyze a single ticket with semaphore-controlled concurrency."""
    async with _ANALYSIS_SEMAPHORE:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, run_analysis, ticket)


async def _analyze_tickets_batched(tickets: List[str]) -> List[AnalysisResponse]:
    """
    Analyze tickets in controlled batches instead of firing all at once.
    Uses a semaphore to limit concurrent LLM calls and processes in waves.
    """
    results: List[AnalysisResponse] = []

    for i in range(0, len(tickets), _BATCH_SIZE):
        batch = tickets[i:i + _BATCH_SIZE]
        batch_tasks = [_analyze_single(ticket) for ticket in batch]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

        for result in batch_results:
            if isinstance(result, Exception):
                logger.error(f"Analysis failed for ticket in batch {i // _BATCH_SIZE}: {result}")
                # Create a fallback result for failed analyses
                results.append(AnalysisResponse(
                    category="REQUEST",
                    priority="MEDIUM",
                    sentiment="Neutral",
                    keywords="analysis-failed",
                    confidenceScore=0.0,
                ))
            else:
                results.append(result)

    return results


# ---------------------------------------------------------------------------
# SSE streaming helper
# ---------------------------------------------------------------------------

def _sse_event(data: dict, event: str = "message") -> str:
    """Format a single Server-Sent Event."""
    payload = json.dumps(data, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration and monitoring."""
    return {"status": "healthy", "service": "ticket-ai-analysis"}


@app.post("/analyze-ticket", response_model=List[AnalyzedTicket])
async def analyze_ticket_file(file: UploadFile = File(...)):
    """
    Endpoint to upload a file (txt, csv, docx, pdf), detect multiple tickets,
    run the multi-agent analysis on each ticket with BATCHED concurrency,
    and return all results with text and analysis.
    """
    try:
        content = await file.read()
        is_list, parsed_content = extract_tickets_or_text(file.filename, content)

        tickets: List[str] = []
        if is_list:
            tickets = parsed_content
        else:
            if not parsed_content.strip():
                raise HTTPException(status_code=400, detail="Could not extract text from the file.")
            # Use the splitter agent to detect and split multiple tickets
            loop = asyncio.get_event_loop()
            tickets = await loop.run_in_executor(_executor, split_tickets, parsed_content)

        if not tickets:
            raise HTTPException(status_code=400, detail="No tickets found to analyze.")

        # Analyze tickets with batched concurrency (not all at once)
        analyses = await _analyze_tickets_batched(tickets)

        # Combine ticket text with analysis
        analyzed_tickets = [
            AnalyzedTicket(text=ticket, analysis=analysis)
            for ticket, analysis in zip(tickets, analyses)
        ]

        return analyzed_tickets

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-and-create-tickets")
async def analyze_and_create_tickets(
    file: UploadFile = File(...),
    jwt_token: str = Form(...),
    source: str = Form(...),
):
    """
    SSE streaming endpoint: upload a file, analyze tickets in batches,
    create each in the backend, and stream real-time progress events.

    Events sent:
      - event: progress  → { current, total, status, ticket? }
      - event: complete  → { status, successCount, failCount, errors }
      - event: error     → { status, message }
    """

    # --- Phase 1: Parse the file (fast, synchronous-ish) ---
    try:
        content = await file.read()
        is_list, parsed_content = extract_tickets_or_text(file.filename, content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {e}")

    async def event_stream():
        tickets: List[str] = []

        try:
            # --- Phase 2: Split into tickets ---
            if is_list:
                tickets = parsed_content
            else:
                if not parsed_content.strip():
                    yield _sse_event({"status": "error", "message": "Could not extract text from the file."}, "error")
                    return
                loop = asyncio.get_event_loop()
                tickets = await loop.run_in_executor(_executor, split_tickets, parsed_content)

            if not tickets:
                yield _sse_event({"status": "error", "message": "No tickets found to analyze."}, "error")
                return

            total = len(tickets)
            yield _sse_event({"status": "started", "total": total}, "progress")

            # --- Phase 3: Process tickets in batches ---
            backend = BackendClient(jwt_token=jwt_token)
            success_count = 0
            fail_count = 0
            errors = []

            for i in range(0, total, _BATCH_SIZE):
                batch = tickets[i:i + _BATCH_SIZE]
                batch_indices = list(range(i, min(i + len(batch), total)))

                # Analyze the batch concurrently
                batch_tasks = [_analyze_single(ticket) for ticket in batch]
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

                # Create tickets in backend for this batch
                for j, (ticket_text, analysis_result) in enumerate(zip(batch, batch_results)):
                    current_index = batch_indices[j] + 1  # 1-based for display

                    if isinstance(analysis_result, Exception):
                        fail_count += 1
                        error_msg = f"Ticket {current_index}: Analysis failed — {str(analysis_result)}"
                        errors.append(error_msg)
                        logger.error(error_msg)
                        yield _sse_event({
                            "status": "ticket_failed",
                            "current": current_index,
                            "total": total,
                            "error": error_msg,
                        }, "progress")
                        continue

                    # Build the payload and create in backend
                    try:
                        ticket_payload = {
                            "title": analysis_result.category.value,
                            "description": ticket_text,
                            "source": source.upper(),
                            "aiAnalysis": {
                                "category": analysis_result.category.value,
                                "priority": analysis_result.priority.value,
                                "sentiment": analysis_result.sentiment,
                                "keywords": analysis_result.keywords,
                                "confidenceScore": analysis_result.confidenceScore,
                            },
                        }

                        # Run the backend create in the executor to avoid blocking
                        created_ticket = await asyncio.get_event_loop().run_in_executor(
                            _executor, backend.create_ticket, ticket_payload
                        )

                        success_count += 1
                        yield _sse_event({
                            "status": "ticket_created",
                            "current": current_index,
                            "total": total,
                            "ticket": {
                                "id": created_ticket.get("id"),
                                "title": created_ticket.get("title"),
                                "category": analysis_result.category.value,
                                "priority": analysis_result.priority.value,
                            },
                        }, "progress")

                    except Exception as create_err:
                        fail_count += 1
                        error_msg = f"Ticket {current_index}: Backend creation failed — {str(create_err)}"
                        errors.append(error_msg)
                        logger.error(error_msg)

                        # Retry once after a short delay
                        try:
                            await asyncio.sleep(1)
                            created_ticket = await asyncio.get_event_loop().run_in_executor(
                                _executor, backend.create_ticket, ticket_payload
                            )
                            # Retry succeeded — fix counts
                            fail_count -= 1
                            success_count += 1
                            errors.pop()
                            yield _sse_event({
                                "status": "ticket_created",
                                "current": current_index,
                                "total": total,
                                "ticket": {
                                    "id": created_ticket.get("id"),
                                    "title": created_ticket.get("title"),
                                    "category": analysis_result.category.value,
                                    "priority": analysis_result.priority.value,
                                },
                            }, "progress")
                        except Exception:
                            yield _sse_event({
                                "status": "ticket_failed",
                                "current": current_index,
                                "total": total,
                                "error": error_msg,
                            }, "progress")

            # --- Phase 4: Complete ---
            yield _sse_event({
                "status": "complete",
                "successCount": success_count,
                "failCount": fail_count,
                "total": total,
                "errors": errors,
            }, "complete")

        except Exception as e:
            logger.exception("Fatal error during streaming ticket processing")
            yield _sse_event({"status": "error", "message": str(e)}, "error")

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/analyze-text", response_model=List[AnalysisResponse])
async def analyze_ticket_text(request: TextAnalysisRequest):
    """
    Endpoint to directly send text for analysis, detect multiple tickets if present,
    run the multi-agent analysis on each with BATCHED concurrency, and return all results.
    """
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty.")

        loop = asyncio.get_event_loop()
        tickets = await loop.run_in_executor(_executor, split_tickets, request.text)

        if not tickets:
            raise HTTPException(status_code=400, detail="No tickets found to analyze.")

        return await _analyze_tickets_batched(tickets)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/chat", response_model=AgentChatResponse)
async def agent_chat(request: AgentChatRequest):
    """
    Tool-driven support agent. Conversation history is kept client-side only.
    Pass jwtToken so tools can call the Spring backend as the authenticated user.
    """
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor,
            lambda: run_support_agent_chat(
                jwt_token=request.jwtToken,
                is_admin=request.isAdmin,
                history=[h.model_dump() for h in request.history],
                message=request.message,
                confirmed_action=(
                    request.confirmedAction.model_dump() if request.confirmedAction else None
                ),
            ),
        )
        return AgentChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
