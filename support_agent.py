from typing import List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from agent_tools import build_tools, execute_confirmed_action, extract_agent_extras
from backend_client import BackendClient

SYSTEM_PROMPT = """You are SupportAI, the intelligent assistant for the Intelligent Support Platform.

You help users and administrators manage support tickets using your tools. Be concise, professional, and proactive.

Rules:
- Use tools to fetch real data; never invent ticket IDs or statuses.
- When the user describes a problem they need tracked, offer to create a ticket with create_support_ticket (source is always CHAT).
- close_ticket and delete_ticket only prepare an action: the user must confirm via the chat buttons. Clearly explain what will happen and tell them to confirm or cancel.
- For admins, you can see all tickets; for regular users, only their own.
- When listing tickets, highlight priority and status when available.
- If a tool returns an error JSON, explain it plainly and suggest next steps.
"""


def _to_langchain_messages(history: List[dict], user_message: Optional[str]) -> list:
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for item in history:
        role = item.get("role")
        content = item.get("content") or ""
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    if user_message:
        messages.append(HumanMessage(content=user_message))
    return messages


def run_support_agent_chat(
    *,
    jwt_token: str,
    is_admin: bool,
    history: List[dict],
    message: Optional[str] = None,
    confirmed_action: Optional[dict] = None,
) -> dict:
    client = BackendClient(jwt_token)

    if confirmed_action:
        action = confirmed_action.get("action")
        ticket_id = confirmed_action.get("ticketId") or confirmed_action.get("ticket_id")
        if not action or ticket_id is None:
            return {
                "reply": "Invalid confirmation payload.",
                "tickets": [],
                "pendingConfirmation": None,
                "toolTraces": [],
            }
        try:
            result = execute_confirmed_action(client, action, int(ticket_id))
            if action == "close_ticket":
                reply = f"Ticket #{ticket_id} has been closed."
            else:
                reply = f"Ticket #{ticket_id} has been deleted."
            tickets = [result["ticket"]] if result.get("ticket") else []
            if action == "delete_ticket":
                tickets = []
            return {
                "reply": reply,
                "tickets": tickets,
                "pendingConfirmation": None,
                "toolTraces": [{"tool": action, "label": "Confirmed action executed"}],
            }
        except Exception as e:
            return {
                "reply": f"Could not complete the action: {e}",
                "tickets": [],
                "pendingConfirmation": None,
                "toolTraces": [],
            }

    if not message or not message.strip():
        return {
            "reply": "Please send a message.",
            "tickets": [],
            "pendingConfirmation": None,
            "toolTraces": [],
        }

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
    tools = build_tools(client, is_admin)
    agent = create_react_agent(llm, tools)

    lc_messages = _to_langchain_messages(history, message.strip())
    result = agent.invoke({"messages": lc_messages})
    out_messages = result.get("messages", [])

    reply = ""
    for msg in reversed(out_messages):
        if isinstance(msg, AIMessage) and msg.content:
            reply = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    tickets, pending, traces = extract_agent_extras(out_messages)

    # Extract tickets message if no tickets found
    tickets_message = None
    if not tickets and len(out_messages) > 0:
        for msg in reversed(out_messages):
            if hasattr(msg, 'content') and isinstance(msg.content, str):
                try:
                    import json
                    data = json.loads(msg.content)
                    if data.get("message") and data.get("tickets") == []:
                        tickets_message = data.get("message")
                        break
                except:
                    pass

    return {
        "reply": reply or "Done.",
        "tickets": tickets,
        "ticketsMessage": tickets_message,
        "pendingConfirmation": pending,
        "toolTraces": traces,
    }
