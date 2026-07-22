import os

from dotenv import load_dotenv

load_dotenv()

import asyncio

from functools import lru_cache

from typing import TypedDict, List, Optional

from pydantic import BaseModel, Field

from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, END

from models import AnalysisResponse, TicketPriority, TicketCategory



# Internal Pydantic models for structured agent outputs

class PreprocessorResult(BaseModel):

    summary: str = Field(description="A concise summary of the support ticket, highlighting the main issue.")

    detected_language: str = Field(description="The primary language of the ticket (e.g. English, French, Spanish).")



class ClassificationResult(BaseModel):

    category: TicketCategory = Field(description="The matching category for the ticket.")

    priority: TicketPriority = Field(description="The calculated priority level based on impact and urgency.")



class SentimentResult(BaseModel):

    sentiment: str = Field(description="The customer sentiment (e.g. Frustrated, Neutral, Happy, Angry, Positive - max 50 chars).")

    keywords: str = Field(description="Comma-separated key phrases or keywords related to the issue (max 1000 chars).")



class ValidationResult(BaseModel):

    is_valid: bool = Field(description="True if the classification and details are logical, consistent and match backend constraints.")

    feedback: Optional[str] = Field(description="If invalid, detailed guidance on why the classification is incorrect and how to fix it.")

    confidence_score: float = Field(description="Confidence score between 0.0 and 1.0 reflecting how clear the ticket issue is.")



class TicketSplits(BaseModel):

    is_multiple: bool = Field(description="True if the text contains multiple separate customer support tickets or messages.")

    tickets: List[str] = Field(description="A list of individual ticket texts extracted from the input. If it is only a single ticket, return a list with just the single ticket text.")



# LLM Singleton — Created ONCE at module import, reused across all requests

_LLM = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)



# Pre-built chains — Constructed ONCE at module load, fully stateless & reusable

_PREPROCESS_CHAIN = (

    ChatPromptTemplate.from_messages([

        ("system", "You are the Preprocessor Agent. Your task is to analyze the raw support ticket. "

                   "Provide a concise summary highlighting the customer's core problem, and detect the language."),

        ("human", "Ticket content:\n\n{ticket_text}")

    ])

    | _LLM.with_structured_output(PreprocessorResult)

)



_SENTIMENT_CHAIN = (

    ChatPromptTemplate.from_messages([

        ("system", "You are the Sentiment & Keywords Agent. Analyze the support ticket and extract:\n"

                   "1. Sentiment: The customer's emotional tone (e.g. Frustrated, Happy, Calm, Upset, Neutral, Positive, Negative - max 50 chars).\n"

                   "2. Keywords: Comma-separated list of keywords representing the core technical components, products, or subjects discussed (max 1000 chars)."),

        ("human", "Ticket content:\n\n{ticket_text}")

    ])

    | _LLM.with_structured_output(SentimentResult)

)



_CLASSIFY_PROMPT = ChatPromptTemplate.from_messages([

    ("system", "You are the Classifier Agent. Your task is to classify the support ticket into a Category and Priority.\n"

               "Categories:\n"

               "- TECHNICAL: Technical bugs, application crashes, server errors, API issues, integration problems.\n"

               "- BILLING: Invoice issues, payment failures, refunds, price plans, subscription renewals.\n"

               "- ACCOUNT: Login issues, registration, password resets, account deletion, permission changes.\n"

               "- COMPLAINT: Customer expressing anger, poor service, dissatisfaction with policy.\n"

               "- REQUEST: Feature requests, asking for documentation, general questions, feedback.\n\n"

               "Priorities:\n"

               "- HIGH: Critical system outages, payment blockages, loss of core functionality.\n"

               "- MEDIUM: Important issues with manual workarounds, minor billing issues, account access delays.\n"

               "- LOW: Non-blocking queries, feature requests, minor styling bugs.\n"

               "{feedback_context}"),

    ("human", "Ticket Summary: {summary}\nOriginal Ticket:\n{ticket_text}")

])

_CLASSIFY_CHAIN = _CLASSIFY_PROMPT | _LLM.with_structured_output(ClassificationResult)



_VALIDATE_CHAIN = (

    ChatPromptTemplate.from_messages([

        ("system", "You are the Validator & Supervisor Agent. Inspect the classification results for logical consistency.\n"

                   "Ensure that a billing or invoice ticket is classified as BILLING (not TECHNICAL).\n"

                   "Ensure that user login issues are classified as ACCOUNT.\n"

                   "Ensure that critical downtime is flagged as HIGH priority.\n"

                   "Evaluate and provide a realistic confidence score (0.0 to 1.0) for the analysis based on clarity."),

        ("human", "Original Ticket: {ticket_text}\n"

                  "Proposed Category: {category}\n"

                  "Proposed Priority: {priority}\n"

                  "Proposed Sentiment: {sentiment}")

    ])

    | _LLM.with_structured_output(ValidationResult)

)



_SPLITTER_CHAIN = (

    ChatPromptTemplate.from_messages([

        ("system", "You are the Ticket Splitter Agent. Examine the provided text. "

                   "Determine if it contains a single customer support ticket or multiple separate customer support tickets/requests/messages. "

                   "Split the input into individual ticket texts, keeping each message intact. "

                   "If the text only contains a single ticket or a single logical issue, return is_multiple=False and list the entire text in the tickets array."),

        ("human", "Input text:\n\n{text}")

    ])

    | _LLM.with_structured_output(TicketSplits)

)



# State definition

class AgentState(TypedDict):

    ticket_text: str

    summary: Optional[str]

    detected_language: Optional[str]

    category: Optional[TicketCategory]

    priority: Optional[TicketPriority]

    sentiment: Optional[str]

    keywords: Optional[str]

    confidence_score: Optional[float]

    attempts: int

    validation_errors: Optional[str]

    final_result: Optional[AnalysisResponse]



# Node 1: Preprocessor Agent

def preprocessor_agent(state: AgentState) -> dict:

    result = _PREPROCESS_CHAIN.invoke({"ticket_text": state["ticket_text"]})

    return {

        "summary": result.summary,

        "detected_language": result.detected_language

    }



# Node 2: Classifier Agent — Accepts dynamic feedback

def classifier_agent(state: AgentState) -> dict:

    feedback_context = ""

    if state.get("validation_errors"):

        feedback_context = (

            f"\n[PREVIOUS CLASSIFICATION FAILURE - CORRECTION REQUIRED]\n"

            f"The previous classification was rejected with the following feedback:\n"

            f"{state['validation_errors']}\n"

        )



    result = _CLASSIFY_CHAIN.invoke({

        "summary": state["summary"],

        "ticket_text": state["ticket_text"],

        "feedback_context": feedback_context

    })



    return {

        "category": result.category,

        "priority": result.priority,

        "attempts": state.get("attempts", 0) + 1

    }



# Node 3: Sentiment & Keywords Analyst Agent

def sentiment_agent(state: AgentState) -> dict:

    result = _SENTIMENT_CHAIN.invoke({"ticket_text": state["ticket_text"]})

    return {

        "sentiment": result.sentiment,

        "keywords": result.keywords

    }



# Node 4: Validator & Supervisor Agent

def validator_agent(state: AgentState) -> dict:

    result = _VALIDATE_CHAIN.invoke({

        "ticket_text": state["ticket_text"],

        "category": state["category"].value if state["category"] else "None",

        "priority": state["priority"].value if state["priority"] else "None",

        "sentiment": state["sentiment"]

    })



    if not result.is_valid and state["attempts"] < 3:

        return {

            "validation_errors": result.feedback,

            "confidence_score": result.confidence_score

        }

    else:

        final_response = AnalysisResponse(

            category=state["category"],

            priority=state["priority"],

            sentiment=state["sentiment"],

            keywords=state["keywords"],

            confidenceScore=result.confidence_score

        )

        return {

            "validation_errors": None,

            "confidence_score": result.confidence_score,

            "final_result": final_response

        }



# Conditional Router

def should_continue(state: AgentState):

    if state.get("validation_errors") and state.get("attempts", 0) < 3:

        return "classify"

    return END



# Build & Compile LangGraph

workflow = StateGraph(AgentState)

workflow.add_node("preprocess", preprocessor_agent)

workflow.add_node("classify", classifier_agent)

workflow.add_node("sentiment", sentiment_agent)

workflow.add_node("validate", validator_agent)



workflow.set_entry_point("preprocess")

workflow.add_edge("preprocess", "classify")

workflow.add_edge("preprocess", "sentiment")

workflow.add_edge("classify", "validate")

workflow.add_edge("sentiment", "validate")

workflow.add_conditional_edges(

    "validate",

    should_continue,

    {"classify": "classify", END: END}

)



# Compiled once — reused for every analysis request

app_graph = workflow.compile()



# Public API

def split_tickets(text: str) -> List[str]:

    """

    Examines the text with a pre-built chain and splits into a list of tickets.

    """

    result = _SPLITTER_CHAIN.invoke({"text": text})

    return result.tickets



def run_analysis(ticket_text: str) -> AnalysisResponse:

    """

    Runs the full multi-agent analysis pipeline on a single ticket.

    The graph is pre-compiled and chains are pre-built — no cold-start on each call.

    """

    initial_state = {

        "ticket_text": ticket_text,

        "summary": None,

        "detected_language": None,

        "category": None,

        "priority": None,

        "sentiment": None,

        "keywords": None,

        "confidence_score": None,

        "attempts": 0,

        "validation_errors": None,

        "final_result": None

    }

    result_state = app_graph.invoke(initial_state)

    return result_state["final_result"]

