from enum import Enum
from pydantic import BaseModel, Field

class TicketPriority(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

class TicketCategory(str, Enum):
    TECHNICAL = "TECHNICAL"
    BILLING = "BILLING"
    ACCOUNT = "ACCOUNT"
    COMPLAINT = "COMPLAINT"
    REQUEST = "REQUEST"

class AnalysisResponse(BaseModel):
    category: TicketCategory = Field(description="The category of the ticket")
    priority: TicketPriority = Field(description="The priority of the ticket")
    sentiment: str = Field(description="The sentiment of the ticket (max 50 characters, e.g., 'Positive', 'Negative', 'Neutral', 'Frustrated')")
    keywords: str = Field(description="Comma-separated keywords extracted from the ticket (max 1000 characters)")
    confidenceScore: float = Field(description="The confidence score of the analysis between 0.0 and 1.0")
