import json

from typing import Any, List, Optional



from langchain_core.tools import StructuredTool

from pydantic import BaseModel, Field



from backend_client import BackendClient

from ai_agent import run_analysis





PRIORITY_RANK = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}





def _latest_priority(ticket: dict) -> Optional[str]:

    analyses = ticket.get("analyses") or []

    if not analyses:

        return None

    latest = max(analyses, key=lambda a: a.get("createdAt") or "")

    p = latest.get("priority")

    return p.value if hasattr(p, "value") else p





def _summarize_ticket(ticket: dict) -> dict:

    analyses = ticket.get("analyses") or []

    latest = max(analyses, key=lambda a: a.get("createdAt") or "", default=None) if analyses else None

    return {

        "id": ticket.get("id"),

        "title": ticket.get("title"),

        "status": ticket.get("status"),

        "source": ticket.get("source"),

        "createdAt": ticket.get("createdAt"),

        "userEmail": ticket.get("userEmail"),

        "priority": (latest or {}).get("priority"),

        "category": (latest or {}).get("category"),

        "sentiment": (latest or {}).get("sentiment"),

    }





def _sort_by_importance(tickets: List[dict]) -> List[dict]:

    def key(t: dict):

        p = _latest_priority(t)

        rank = PRIORITY_RANK.get(p, 3)

        return (rank, t.get("createdAt") or "")



    return sorted(tickets, key=key)





class ListOpenTicketsInput(BaseModel):

    limit: int = Field(default=10, ge=1, le=50, description="Maximum number of tickets to return")



class ListClosedTicketsInput(BaseModel):

    limit: int = Field(default=10, ge=1, le=50, description="Maximum number of tickets to return")



class FilterTicketsInput(BaseModel):

    status: Optional[str] = Field(default=None, description="Filter by status (OPEN, CLOSED, IN_PROGRESS)")

    priority: Optional[str] = Field(default=None, description="Filter by AI priority (HIGH, MEDIUM, LOW)")

    category: Optional[str] = Field(default=None, description="Filter by AI category (TECHNICAL, BILLING, ACCOUNT, COMPLAINT, REQUEST)")

    sentiment: Optional[str] = Field(default=None, description="Filter by sentiment (POSITIVE, NEGATIVE, NEUTRAL, FRUSTRATED)")

    source: Optional[str] = Field(default=None, description="Filter by source (WEB, CHAT, CSV)")

    user_email: Optional[str] = Field(default=None, description="Filter by user email address")

    title_query: Optional[str] = Field(default=None, description="Search text matched against ticket titles")

    limit: int = Field(default=10, ge=1, le=50, description="Maximum number of tickets to return")



class ListTicketsByUserInput(BaseModel):

    user_email: str = Field(description="The user email to filter tickets by")

    limit: int = Field(default=10, ge=1, le=50, description="Maximum number of tickets to return")





class GetTicketInput(BaseModel):

    ticket_id: int = Field(description="The ticket ID")





class CreateTicketInput(BaseModel):

    title: str = Field(min_length=3, description="Short title for the ticket")

    description: str = Field(min_length=10, description="Detailed description of the issue")





class TicketIdInput(BaseModel):

    ticket_id: int = Field(description="The ticket ID to act on")





class SearchTicketsInput(BaseModel):

    title_query: str = Field(description="Search text matched against ticket titles")

    limit: int = Field(default=10, ge=1, le=30)





def build_tools(client: BackendClient, is_admin: bool) -> List[StructuredTool]:

    def list_open_tickets(limit: int = 10) -> str:

        """List open tickets for the current user (or all tickets if admin)."""

        page = client.list_tickets(admin=is_admin, status="OPEN", size=min(limit, 50))

        items = [_summarize_ticket(t) for t in page.get("content", [])]

        if not items:
            return json.dumps({"tickets": [], "total": 0, "message": "No open tickets found"})

        return json.dumps({"tickets": items, "total": page.get("totalElements", len(items))})



    def list_closed_tickets(limit: int = 10) -> str:

        """List closed tickets for the current user (or all tickets if admin)."""

        page = client.list_tickets(admin=is_admin, status="CLOSED", size=min(limit, 50))

        items = [_summarize_ticket(t) for t in page.get("content", [])]

        if not items:
            return json.dumps({"tickets": [], "total": 0, "message": "No closed tickets found"})

        return json.dumps({"tickets": items, "total": page.get("totalElements", len(items))})



    def get_important_open_tickets(limit: int = 10) -> str:

        """List the most important open tickets, ranked by AI priority (HIGH first)."""

        page = client.list_tickets(admin=is_admin, status="OPEN", size=50)

        ranked = _sort_by_importance(page.get("content", []))[:limit]

        items = [_summarize_ticket(t) for t in ranked]

        if not items:
            return json.dumps({"tickets": [], "total": 0, "message": "No important open tickets found"})

        return json.dumps({"tickets": items, "total": page.get("totalElements", len(items))})



    def get_ticket_details(ticket_id: int) -> str:

        """Get full details for a single ticket by ID."""

        ticket = client.get_ticket(ticket_id)

        return json.dumps(_summarize_ticket(ticket) | {"description": ticket.get("description")})



    def search_tickets(title_query: str, limit: int = 10) -> str:

        """Search tickets by title keyword."""

        page = client.list_tickets(admin=is_admin, title=title_query, size=min(limit, 30))

        items = [_summarize_ticket(t) for t in page.get("content", [])]

        if not items:
            return json.dumps({"tickets": [], "total": 0, "query": title_query, "message": f"No tickets found matching '{title_query}'"})

        return json.dumps({"tickets": items, "query": title_query})



    def filter_tickets(
        status: Optional[str] = None,
        priority: Optional[str] = None,
        category: Optional[str] = None,
        sentiment: Optional[str] = None,
        source: Optional[str] = None,
        user_email: Optional[str] = None,
        title_query: Optional[str] = None,
        limit: int = 10
    ) -> str:
        """Filter tickets by multiple criteria (status, priority, category, sentiment, source, user email, title). All filters are AND logic - ticket must match all provided criteria."""

        page = client.list_tickets(
            admin=is_admin,
            status=status,
            title=title_query,
            size=min(limit, 50)
        )

        items = page.get("content", [])

        # Apply additional filters that backend doesn't support
        if priority or category or sentiment or source or user_email:
            filtered = []
            for ticket in items:
                analyses = ticket.get("analyses") or []
                latest = max(analyses, key=lambda a: a.get("createdAt") or "", default=None) if analyses else None

                # Check priority
                if priority:
                    p = (latest or {}).get("priority")
                    p_val = p.value if hasattr(p, "value") else p
                    if p_val != priority:
                        continue

                # Check category
                if category:
                    c = (latest or {}).get("category")
                    c_val = c.value if hasattr(c, "value") else c
                    if c_val != category:
                        continue

                # Check sentiment
                if sentiment:
                    s = (latest or {}).get("sentiment")
                    if s != sentiment:
                        continue

                # Check source
                if source:
                    s = ticket.get("source")
                    s_val = s.value if hasattr(s, "value") else s
                    if s_val != source:
                        continue

                # Check user email
                if user_email:
                    u = ticket.get("userEmail")
                    if u != user_email:
                        continue

                filtered.append(ticket)

            items = filtered[:limit]

        if not items:
            return json.dumps({"tickets": [], "total": 0, "message": "No tickets found matching the specified criteria"})

        return json.dumps({"tickets": [_summarize_ticket(t) for t in items], "total": len(items)})



    def list_tickets_by_user(user_email: str, limit: int = 10) -> str:
        """List all tickets for a specific user by email address."""

        page = client.list_tickets(admin=is_admin, size=min(limit, 50))
        items = page.get("content", [])

        filtered = [t for t in items if t.get("userEmail") == user_email][:limit]

        if not filtered:
            return json.dumps({"tickets": [], "total": 0, "message": f"No tickets found for user '{user_email}'"})

        return json.dumps({"tickets": [_summarize_ticket(t) for t in filtered], "total": len(filtered)})



    def create_support_ticket(title: str, description: str) -> str:

        """Create a new support ticket with source CHAT. AI analysis is always performed."""

        payload: dict[str, Any] = {

            "title": title,

            "description": description,

            "source": "CHAT",

        }

        try:

            analysis = run_analysis(description)

            payload["aiAnalysis"] = {

                "category": analysis.category.value,

                "priority": analysis.priority.value,

                "sentiment": analysis.sentiment,

                "keywords": analysis.keywords,

                "confidenceScore": analysis.confidenceScore,

            }

        except Exception:

            pass

        created = client.create_ticket(payload)

        return json.dumps({"created": True, "ticket": _summarize_ticket(created)})



    def close_ticket(ticket_id: int) -> str:

        """Request to close a ticket (sets status to CLOSED). Requires user confirmation in the chat UI."""

        try:

            ticket = client.get_ticket(ticket_id)

        except Exception as e:

            return json.dumps({"error": str(e)})

        return json.dumps(

            {

                "requires_confirmation": True,

                "action": "close_ticket",

                "ticket_id": ticket_id,

                "ticket_title": ticket.get("title"),

                "message": f"Close ticket #{ticket_id} « {ticket.get('title')} »?",

            }

        )



    def delete_ticket(ticket_id: int) -> str:

        """Request to permanently delete a ticket. Requires user confirmation in the chat UI."""

        try:

            ticket = client.get_ticket(ticket_id)

        except Exception as e:

            return json.dumps({"error": str(e)})

        return json.dumps(

            {

                "requires_confirmation": True,

                "action": "delete_ticket",

                "ticket_id": ticket_id,

                "ticket_title": ticket.get("title"),

                "message": f"Permanently delete ticket #{ticket_id} « {ticket.get('title')} »?",

            }

        )



    return [

        StructuredTool.from_function(

            func=list_open_tickets,

            name="list_open_tickets",

            description="List open support tickets.",

            args_schema=ListOpenTicketsInput,

        ),

        StructuredTool.from_function(

            func=list_closed_tickets,

            name="list_closed_tickets",

            description="List closed support tickets.",

            args_schema=ListClosedTicketsInput,

        ),

        StructuredTool.from_function(

            func=get_important_open_tickets,

            name="get_important_open_tickets",

            description="Get the highest-priority open tickets (HIGH priority first).",

            args_schema=ListOpenTicketsInput,

        ),

        StructuredTool.from_function(

            func=get_ticket_details,

            name="get_ticket_details",

            description="Get details of one ticket by ID.",

            args_schema=GetTicketInput,

        ),

        StructuredTool.from_function(

            func=search_tickets,

            name="search_tickets",

            description="Search tickets by title.",

            args_schema=SearchTicketsInput,

        ),

        StructuredTool.from_function(

            func=filter_tickets,

            name="filter_tickets",

            description="Filter tickets by multiple criteria (status, priority, category, sentiment, source, user email, title). All filters are AND logic - ticket must match all provided criteria.",

            args_schema=FilterTicketsInput,

        ),

        StructuredTool.from_function(

            func=list_tickets_by_user,

            name="list_tickets_by_user",

            description="List all tickets for a specific user by email address.",

            args_schema=ListTicketsByUserInput,

        ),

        StructuredTool.from_function(

            func=create_support_ticket,

            name="create_support_ticket",

            description="Create a ticket from the chat (source CHAT) with AI analysis. Use when the user reports an issue.",

            args_schema=CreateTicketInput,

        ),

        StructuredTool.from_function(

            func=close_ticket,

            name="close_ticket",

            description="Close a ticket. Does not run until the user confirms in the UI.",

            args_schema=TicketIdInput,

        ),

        StructuredTool.from_function(

            func=delete_ticket,

            name="delete_ticket",

            description="Delete a ticket permanently. Does not run until the user confirms in the UI.",

            args_schema=TicketIdInput,

        ),

    ]





def execute_confirmed_action(client: BackendClient, action: str, ticket_id: int) -> dict:

    if action == "close_ticket":

        ticket = client.get_ticket(ticket_id)

        status = ticket.get("status")

        if hasattr(status, "value"):

            status = status.value

        source = ticket.get("source")

        if hasattr(source, "value"):

            source = source.value

        updated = client.update_ticket(

            ticket_id,

            {

                "title": ticket["title"],

                "description": ticket["description"],

                "source": source,

                "status": "CLOSED",

            },

        )

        return {"success": True, "action": action, "ticket": _summarize_ticket(updated)}

    if action == "delete_ticket":

        client.delete_ticket(ticket_id)

        return {"success": True, "action": action, "ticket_id": ticket_id}

    raise ValueError(f"Unknown action: {action}")





def extract_agent_extras(messages: list) -> tuple[list, Optional[dict], list]:

    """Parse tool outputs for ticket cards, pending confirmation, and tool trace labels."""

    tickets: list = []

    pending: Optional[dict] = None

    traces: list = []



    tool_labels = {

        "list_open_tickets": "Listing open tickets",

        "list_closed_tickets": "Listing closed tickets",

        "get_important_open_tickets": "Ranking important tickets",

        "get_ticket_details": "Fetching ticket details",

        "search_tickets": "Searching tickets",

        "filter_tickets": "Filtering tickets",

        "list_tickets_by_user": "Listing tickets by user",

        "create_support_ticket": "Creating ticket",

        "close_ticket": "Preparing close request",

        "delete_ticket": "Preparing delete request",

    }



    for msg in messages:

        msg_type = getattr(msg, "type", None) or msg.__class__.__name__

        if msg_type == "tool" or msg.__class__.__name__ == "ToolMessage":

            name = getattr(msg, "name", None) or ""

            if name in tool_labels:

                traces.append({"tool": name, "label": tool_labels[name]})

            content = getattr(msg, "content", "") or ""

            if isinstance(content, str):

                try:

                    data = json.loads(content)

                except json.JSONDecodeError:

                    continue

                if data.get("requires_confirmation"):

                    pending = {

                        "action": data.get("action"),

                        "ticketId": data.get("ticket_id"),

                        "ticketTitle": data.get("ticket_title"),

                        "message": data.get("message"),

                    }

                if "tickets" in data and isinstance(data["tickets"], list):

                    tickets = data["tickets"]

                if data.get("created") and data.get("ticket"):

                    tickets = [data["ticket"]]

                if data.get("success") and data.get("ticket"):

                    tickets = [data["ticket"]]



    return tickets, pending, traces

