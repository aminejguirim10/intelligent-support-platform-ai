from typing import Any, List, Optional

from pydantic import BaseModel, Field


class ChatHistoryItem(BaseModel):
    role: str = Field(description="user or assistant")
    content: str


class ConfirmedAction(BaseModel):
    action: str = Field(description="close_ticket or delete_ticket")
    ticketId: int


class AgentChatRequest(BaseModel):
    message: Optional[str] = None
    history: List[ChatHistoryItem] = Field(default_factory=list)
    jwtToken: str
    isAdmin: bool = False
    confirmedAction: Optional[ConfirmedAction] = None


class PendingConfirmation(BaseModel):
    action: str
    ticketId: int
    ticketTitle: Optional[str] = None
    message: Optional[str] = None


class ToolTrace(BaseModel):
    tool: str
    label: str


class AgentTicketCard(BaseModel):
    id: Optional[int] = None
    title: Optional[str] = None
    status: Optional[Any] = None
    source: Optional[Any] = None
    priority: Optional[Any] = None
    category: Optional[Any] = None
    sentiment: Optional[str] = None
    userEmail: Optional[str] = None
    createdAt: Optional[str] = None


class AgentChatResponse(BaseModel):
    reply: str
    tickets: List[AgentTicketCard] = Field(default_factory=list)
    pendingConfirmation: Optional[PendingConfirmation] = None
    toolTraces: List[ToolTrace] = Field(default_factory=list)
