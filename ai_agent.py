import os
import re

from dotenv import load_dotenv

load_dotenv()

from typing import List

from pydantic import BaseModel, Field

from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate

from models import AnalysisResponse, TicketPriority, TicketCategory



# LLM Singleton — Created ONCE at module import, reused across all requests

_LLM = ChatGroq(model="openai/gpt-oss-120b", temperature=0.0)



# Single unified analysis model

class UnifiedAnalysisResult(BaseModel):

    category: TicketCategory = Field(description="The matching category for the ticket (TECHNICAL, BILLING, ACCOUNT, COMPLAINT, REQUEST)")

    priority: TicketPriority = Field(description="The calculated priority level (HIGH, MEDIUM, LOW)")

    sentiment: str = Field(description="The customer sentiment (max 50 chars, e.g. Frustrated, Neutral, Happy, Angry, Positive)")

    keywords: str = Field(description="Comma-separated key phrases or keywords related to the issue (max 1000 chars)")

    confidence_score: float = Field(description="Confidence score between 0.0 and 1.0 reflecting how clear the ticket issue is")

    summary: str = Field(description="A concise summary of the support ticket, highlighting the main issue")

    detected_language: str = Field(description="The primary language of the ticket (e.g. English, French, Spanish)")

    advice: str = Field(description="Helpful advice in English on how to resolve the ticket or guide the user (max 500 characters)")



# Single unified prompt that extracts all information in one LLM call

_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([

    ("system", """You are an expert support ticket analyst. Analyze the support ticket and extract ALL of the following information in a single response:

CATEGORIES:
- TECHNICAL: Technical bugs, application crashes, server errors, API issues, integration problems
- BILLING: Invoice issues, payment failures, refunds, price plans, subscription renewals
- ACCOUNT: Login issues, registration, password resets, account deletion, permission changes
- COMPLAINT: Customer expressing anger, poor service, dissatisfaction with policy
- REQUEST: Feature requests, asking for documentation, general questions, feedback

PRIORITIES:
- HIGH: Critical system outages, payment blockages, loss of core functionality
- MEDIUM: Important issues with manual workarounds, minor billing issues, account access delays
- LOW: Non-blocking queries, feature requests, minor styling bugs

SENTIMENT: The customer's emotional tone (max 50 chars)
KEYWORDS: Comma-separated list of keywords (max 1000 chars)
CONFIDENCE SCORE: 0.0 to 1.0 based on clarity of the issue
SUMMARY: Concise summary highlighting the main issue
LANGUAGE: Primary language of the ticket
ADVICE: Helpful advice in English on how to resolve the ticket or guide the user (max 500 characters). This must ALWAYS be in English regardless of the ticket's language.

Ensure consistency: billing/invoice tickets should be BILLING (not TECHNICAL), login issues should be ACCOUNT, critical downtime should be HIGH priority."""),

    ("human", "Ticket content:\n\n{ticket_text}")

])



# Single unified chain - ONE LLM call per ticket

_ANALYSIS_CHAIN = _ANALYSIS_PROMPT | _LLM.with_structured_output(UnifiedAnalysisResult)



# Ticket splitter for handling multiple tickets in one text

class TicketSplits(BaseModel):

    is_multiple: bool = Field(description="True if the text contains multiple separate customer support tickets or messages.")

    tickets: List[str] = Field(description="A list of individual ticket texts extracted from the input. If it is only a single ticket, return a list with just the single ticket text.")



_SPLITTER_PROMPT = ChatPromptTemplate.from_messages([

    ("system", "You are the Ticket Splitter Agent. Examine the provided text. "

               "Determine if it contains a single customer support ticket or multiple separate customer support tickets/requests/messages. "

               "Split the input into individual ticket texts, keeping each message intact. "

               "If the text only contains a single ticket or a single logical issue, return is_multiple=False and list the entire text in the tickets array."),

    ("human", "Input text:\n\n{text}")

])



_SPLITTER_CHAIN = _SPLITTER_PROMPT | _LLM.with_structured_output(TicketSplits)



# Chunking config for the ticket splitter

_CHUNK_SIZE = 6000       # characters per chunk sent to the splitter LLM

_CHUNK_OVERLAP = 500     # overlap between chunks so tickets at boundaries aren't lost



def _deduplicate_tickets(tickets: List[str]) -> List[str]:

    """Remove near-duplicate tickets that appear in overlapping chunks."""

    seen: List[str] = []

    for ticket in tickets:

        normalized = ticket.strip()

        if not normalized:

            continue

        # Check if this ticket is a substantial substring of one already seen

        is_dup = False

        for existing in seen:

            # If >80% of the shorter string appears in the longer, it's a duplicate

            shorter, longer = (normalized, existing) if len(normalized) <= len(existing) else (existing, normalized)

            if shorter in longer:

                is_dup = True

                break

        if not is_dup:

            seen.append(normalized)

    return seen



def _calculate_confidence_score(ticket_text: str) -> float:

    """

    Calculate confidence from the ticket content itself.

    The LLM still handles categorization and priority, but confidence is derived

    from the amount of detail, specificity, and structure in the ticket so it

    varies per message instead of defaulting to a constant value.

    """

    text = ticket_text.strip()

    if not text:

        return 0.0

    score = 0.45

    lowered = text.lower()

    if len(text) > 700:

        score += 0.22

    elif len(text) > 350:

        score += 0.16

    elif len(text) > 150:

        score += 0.09

    elif len(text) < 30:

        score -= 0.22

    elif len(text) < 80:

        score -= 0.08

    detail_markers = [

        "error", "exception", "failed", "timeout", "crash", "bug",

        "invoice", "payment", "refund", "login", "password", "api",

        "database", "server", "deployment", "account", "subscription",

    ]

    if any(marker in lowered for marker in detail_markers):

        score += 0.10

    if re.search(r"\b\d{2,}\b", text):

        score += 0.05

    if re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text):

        score += 0.05

    sentence_count = len([part for part in re.split(r"[.!?]+", text) if part.strip()])

    if sentence_count >= 4:

        score += 0.10

    elif sentence_count >= 2:

        score += 0.05

    vague_markers = ["something", "maybe", "not sure", "probably", "might be", "idk"]

    if any(marker in lowered for marker in vague_markers):

        score -= 0.12

    word_count = len(text.split())

    if word_count < 5:

        score -= 0.18

    elif word_count < 10:

        score -= 0.08

    if len(text) > 0 and " " not in text and len(set(text)) < 6:

        score = 0.1

    return max(0.0, min(1.0, round(score, 2)))



def split_tickets(text: str) -> List[str]:

    """

    Examines the text with a pre-built chain and splits into a list of tickets.

    For very large texts (>8000 chars), splits the input into overlapping chunks

    and processes each independently to avoid LLM context window issues.

    Results are deduplicated to handle the overlap regions.

    """

    text = text.strip()

    if not text:

        return []

    # Small enough to process in one shot

    if len(text) <= 8000:

        result = _SPLITTER_CHAIN.invoke({"text": text})

        return [t.strip() for t in result.tickets if t.strip()]

    # Large text — process in chunks

    all_tickets: List[str] = []

    start = 0

    while start < len(text):

        end = min(start + _CHUNK_SIZE, len(text))

        # Try to break at a natural boundary (newline) near the end of the chunk

        if end < len(text):

            newline_pos = text.rfind("\n", start + _CHUNK_SIZE - 200, end)

            if newline_pos > start:

                end = newline_pos + 1

        chunk = text[start:end]

        try:

            result = _SPLITTER_CHAIN.invoke({"text": chunk})

            chunk_tickets = [t.strip() for t in result.tickets if t.strip()]

            all_tickets.extend(chunk_tickets)

        except Exception:

            # If the splitter fails on a chunk, treat the whole chunk as one ticket

            all_tickets.append(chunk.strip())

        # Advance with overlap

        start = end - _CHUNK_OVERLAP if end < len(text) else end

    return _deduplicate_tickets(all_tickets)



def run_analysis(ticket_text: str) -> AnalysisResponse:

    """

    Runs the unified analysis pipeline on a single ticket.

    This makes ONLY ONE LLM call per ticket, extracting all information at once.

    """

    result = _ANALYSIS_CHAIN.invoke({"ticket_text": ticket_text})

    calculated_confidence = _calculate_confidence_score(ticket_text)



    return AnalysisResponse(

        category=result.category,

        priority=result.priority,

        sentiment=result.sentiment,

        keywords=result.keywords,

        confidenceScore=calculated_confidence,

        advice=result.advice

    )
