import asyncio

from concurrent.futures import ThreadPoolExecutor

from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException, Form

from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from dotenv import load_dotenv



load_dotenv()



from file_parser import extract_tickets_or_text

from ai_agent import run_analysis, split_tickets

from models import AnalysisResponse, AnalyzedTicket
from agent_models import AgentChatRequest, AgentChatResponse
from support_agent import run_support_agent_chat
from backend_client import BackendClient


app = FastAPI(title="Ticket AI Analysis Service")



# Configure CORS

app.add_middleware(

    CORSMiddleware,

    allow_origins=["http://localhost:4200"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],

)



# Thread pool for CPU/IO bound LLM calls — shared across all requests

_executor = ThreadPoolExecutor(max_workers=8)



from pydantic import BaseModel, Field



class TextAnalysisRequest(BaseModel):
    text: str


async def _analyze_tickets_concurrently(tickets: List[str]) -> List[AnalysisResponse]:

    """

    Runs run_analysis on each ticket concurrently via a thread pool,

    since LangChain chains are synchronous but IO-bound (network calls to Groq).

    """

    loop = asyncio.get_event_loop()

    tasks = [

        loop.run_in_executor(_executor, run_analysis, ticket)

        for ticket in tickets

    ]

    results = await asyncio.gather(*tasks)

    return list(results)



@app.post("/analyze-ticket", response_model=List[AnalyzedTicket])

async def analyze_ticket_file(file: UploadFile = File(...)):

    """

    Endpoint to upload a file (txt, csv, docx, pdf), detect multiple tickets,

    run the multi-agent analysis on each ticket CONCURRENTLY, and return all results with text and analysis.

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



        # Analyze tickets concurrently
        analyses = await _analyze_tickets_concurrently(tickets)

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
    source: str = Form(...)
):
    """
    Endpoint to upload a file, analyze it, and create tickets directly in the backend.
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
            loop = asyncio.get_event_loop()
            tickets = await loop.run_in_executor(_executor, split_tickets, parsed_content)

        if not tickets:
            raise HTTPException(status_code=400, detail="No tickets found to analyze.")

        # Analyze tickets concurrently
        analyses = await _analyze_tickets_concurrently(tickets)

        # Create tickets in backend
        backend = BackendClient(jwt_token=jwt_token)
        created_tickets = []

        for ticket_text, analysis in zip(tickets, analyses):
            ticket_payload = {
                "title": analysis.category.value,
                "description": ticket_text,
                "source": source.upper(),
                "aiAnalysis": {
                    "category": analysis.category.value,
                    "priority": analysis.priority.value,
                    "sentiment": analysis.sentiment,
                    "keywords": analysis.keywords,
                    "confidenceScore": analysis.confidenceScore
                }
            }
            created_ticket = backend.create_ticket(ticket_payload)
            created_tickets.append(created_ticket)

        return {"tickets": created_tickets, "count": len(created_tickets)}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-text", response_model=List[AnalysisResponse])
async def analyze_ticket_text(request: TextAnalysisRequest):

    """

    Endpoint to directly send text for analysis, detect multiple tickets if present,

    run the multi-agent analysis on each CONCURRENTLY, and return all results.

    """

    try:

        if not request.text.strip():

            raise HTTPException(status_code=400, detail="Text cannot be empty.")



        loop = asyncio.get_event_loop()

        tickets = await loop.run_in_executor(_executor, split_tickets, request.text)



        if not tickets:

            raise HTTPException(status_code=400, detail="No tickets found to analyze.")



        return await _analyze_tickets_concurrently(tickets)



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

