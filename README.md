# 🤖 Intelligent Support Platform AI

Intelligent Support Platform AI is a FastAPI service that provides AI-powered ticket analysis and conversational support agent capabilities. It leverages LangChain, LangGraph, and Groq's LLM to analyze support tickets, automatically categorize them, and provide intelligent assistance through a tool-driven chat interface.

## ✨ Key Features

- **AI-Powered Ticket Analysis**: Multi-agent analysis that extracts category, priority, sentiment, keywords, and provides actionable advice.
- **Automatic Ticket Splitting**: Intelligently detects and separates multiple tickets from a single text input or file.
- **Multi-Format File Support**: Parse and analyze tickets from TXT, CSV, DOCX, and PDF files.
- **Batch Processing**: Controlled concurrent analysis with semaphore-based rate limiting and batch processing.
- **Real-Time Streaming**: Server-Sent Events (SSE) for real-time progress updates during bulk ticket creation.
- **Conversational Support Agent**: Tool-driven AI agent for interactive ticket management via chat.
- **Role-Based Access**: Different tool access levels for admins and regular users.
- **Confidence Scoring**: Algorithmic confidence calculation based on ticket content detail and structure.
- **Multi-Language Support**: Detects ticket language and provides advice in English regardless of input language.
- **CORS Configuration**: Configurable CORS for seamless front-end integration.

## 🧰 Technology Stack

- **Python 3.11**
- **FastAPI**: Modern, fast web framework for building APIs
- **Uvicorn**: ASGI server for running FastAPI applications
- **LangChain**: Framework for building LLM-powered applications
- **LangGraph**: Framework for building stateful, multi-actor applications with LLMs
- **LangChain-Groq**: Integration with Groq's high-performance LLM API
- **Groq LLM**: OpenAI/gpt-oss-120b model for inference
- **Pydantic**: Data validation using Python type annotations
- **HTTPX**: Async HTTP client for backend API communication
- **PDFPlumber**: PDF parsing and text extraction
- **python-docx**: Microsoft Word document parsing
- **Pandas**: Data manipulation for CSV processing
- **python-dotenv**: Environment variable management

## 🏗️ Architecture

The project is organized into the following modules:

- **main.py**: FastAPI application with endpoints for ticket analysis, file processing, and agent chat.
- **ai_agent.py**: Core AI analysis logic using LangChain chains for ticket splitting and analysis.
- **support_agent.py**: Conversational agent implementation using LangGraph's ReAct agent pattern.
- **agent_tools.py**: Tool definitions for the support agent (list, search, filter, create, close, delete tickets).
- **backend_client.py**: HTTP client for communicating with the Spring Boot backend API.
- **file_parser.py**: File parsing utilities for multiple document formats.
- **models.py**: Pydantic models for request/response validation and enums.
- **agent_models.py**: Additional Pydantic models for agent-specific requests.

## 💼 Main Business Modules

### 🎫 AI Ticket Analysis

The ticket analysis pipeline uses a single unified LLM call to extract all information:

- **Category Classification**: TECHNICAL, BILLING, ACCOUNT, COMPLAINT, REQUEST
- **Priority Assignment**: HIGH, MEDIUM, LOW based on severity and impact
- **Sentiment Analysis**: Detects customer emotional tone (max 50 characters)
- **Keyword Extraction**: Comma-separated key phrases related to the issue
- **Confidence Scoring**: Algorithmic score (0.0-1.0) based on content detail and structure
- **Language Detection**: Identifies the primary language of the ticket
- **Advice Generation**: English-language resolution guidance regardless of ticket language

### ✂️ Ticket Splitting

For large texts or files containing multiple tickets:

- Automatic detection of single vs. multiple tickets
- Intelligent splitting at natural boundaries
- Chunking for very large texts (>8000 characters) with overlap handling
- Deduplication to handle overlapping chunks
- Maintains ticket integrity and context

### 💬 Conversational Support Agent

A tool-driven AI agent that assists users and admins:

- **Ticket Listing**: List open/closed tickets with optional limits
- **Priority-Based Ranking**: Get important tickets sorted by AI priority
- **Search & Filter**: Search by title, filter by status/priority/category/sentiment/source
- **User-Based Filtering**: List tickets by specific user email
- **Ticket Creation**: Create tickets from chat with automatic AI analysis
- **Ticket Management**: Close and delete tickets with user confirmation
- **Admin vs. User Access**: Different data visibility based on role

### 📁 File Processing

Support for multiple document formats:

- **TXT**: Plain text parsing
- **CSV**: Structured data extraction
- **DOCX**: Microsoft Word document parsing
- **PDF**: PDF text extraction using PDFPlumber

### 🌊 Streaming Processing

For bulk ticket creation from files:

- Server-Sent Events (SSE) for real-time progress updates
- Batch processing with controlled concurrency (max 5 concurrent LLM calls)
- Progress events: started, ticket_created, ticket_failed, complete
- Automatic retry on backend creation failures
- Detailed error reporting and success/failure counts

## 🌐 API Surface

### Health Check
- `GET /health` - Service health check endpoint

### Ticket Analysis
- `POST /analyze-ticket` - Upload file and analyze tickets (returns analysis only)
- `POST /analyze-text` - Analyze text directly (returns analysis only)
- `POST /analyze-and-create-tickets` - Upload file, analyze, and create tickets in backend (SSE streaming)

### Conversational Agent
- `POST /agent/chat` - Interact with the support agent (requires JWT token)

## 📋 API Details

### POST /analyze-ticket

Upload a file and receive AI analysis for all detected tickets.

**Request**: Multipart form data with file
**Response**: List of analyzed tickets with text and analysis

```json
[
  {
    "text": "Ticket content...",
    "analysis": {
      "category": "TECHNICAL",
      "priority": "HIGH",
      "sentiment": "Frustrated",
      "keywords": "server, error, timeout",
      "confidenceScore": 0.85,
      "advice": "Check server logs and restart the service..."
    }
  }
]
```

### POST /analyze-text

Analyze text directly without file upload.

**Request Body**:
```json
{
  "text": "Ticket content to analyze..."
}
```

**Response**: Same as `/analyze-ticket`

### POST /analyze-and-create-tickets

Upload a file, analyze tickets, and create them in the backend with real-time progress updates.

**Request**: Multipart form data
- `file`: File to process
- `jwt_token`: JWT token for backend authentication
- `source`: Ticket source (WEB, CHAT, EMAIL, etc.)

**Response**: Server-Sent Events stream with events:
- `progress`: { status, current, total, ticket?, error? }
- `complete`: { status, successCount, failCount, total, errors }
- `error`: { status, message }

### POST /agent/chat

Interact with the conversational support agent.

**Request Body**:
```json
{
  "jwtToken": "your-jwt-token",
  "isAdmin": false,
  "message": "List my open tickets",
  "history": [
    { "role": "user", "content": "Previous message" },
    { "role": "assistant", "content": "Previous response" }
  ],
  "confirmedAction": null
}
```

**Response**:
```json
{
  "reply": "Here are your open tickets...",
  "tickets": [...],
  "ticketsMessage": null,
  "pendingConfirmation": null,
  "toolTraces": [
    { "tool": "list_open_tickets", "label": "Listing open tickets" }
  ]
}
```

## ⚙️ Configuration

Create a `.env` file in the project root (see `.env.example`):

```env
# AI Service Configuration
GROQ_API_KEY=your_groq_api_key_here

# Backend Service Configuration
BACKEND_URL=http://localhost:8081

# CORS Configuration
CORS_ALLOWED_ORIGINS=http://localhost:4200,http://localhost:3000
```

## ✅ Prerequisites

- Python 3.11 or higher
- Groq API key (obtain from https://console.groq.com/)
- Access to the Spring Boot backend service
- For local development: Virtual environment

## 🚀 Run Locally

### Clone the repository

```bash
git clone https://github.com/aminejguirim10/intelligent-support-platform-ai.git
cd intelligent-support-platform-ai
```

### Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configure environment

```bash
cp .env.example .env
# Edit .env with your actual configuration
```

### Run the service

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- API: http://localhost:8000
- Health check: http://localhost:8000/health
- Interactive API docs: http://localhost:8000/docs

## 🐳 Docker

The repository includes a multi-stage Dockerfile for production deployment.

### Build the image

```bash
docker build -t intelligent-support-platform-ai .
```

### Run the container

```bash
docker run --rm -p 8000:8000 \
  -e GROQ_API_KEY=your_api_key \
  -e BACKEND_URL=http://backend:8081 \
  -e CORS_ALLOWED_ORIGINS=http://localhost:4200 \
  intelligent-support-platform-ai
```

The container includes:
- Multi-stage build for optimized image size
- Non-root user for security
- Health check endpoint
- Configurable PORT environment variable

## 📌 Operational Notes

### Concurrency Control

- **Semaphore limit**: Maximum 5 concurrent LLM analysis calls
- **Batch size**: Processes tickets in batches of 5
- **Thread pool**: 8 workers for CPU/IO-bound operations
- **Retry logic**: Automatic retry on backend creation failures with 1-second delay

### Performance Considerations

- LLM calls are offloaded to a thread pool to avoid blocking the event loop
- Batch processing prevents overwhelming the LLM API
- Chunking for large texts (>8000 chars) with 500-character overlap
- Deduplication algorithm to handle overlapping chunks

### Error Handling

- Graceful fallback for failed analyses (returns default values)
- Detailed error messages in SSE stream
- Tool execution errors are caught and reported to the user
- HTTP timeout set to 30 seconds for backend calls

### Security

- JWT token required for all backend API calls
- CORS configured to allow specific origins only
- Non-root user in Docker container
- Environment variables for sensitive configuration

## 🔒 Security Best Practices

- Never commit `.env` file with real API keys
- Use environment variables for all sensitive configuration
- Rotate Groq API keys regularly
- Implement rate limiting in production
- Monitor LLM API usage and costs
- Validate all file uploads and limit file sizes

## 📊 Ticket Categories

- **TECHNICAL**: Bugs, crashes, server errors, API issues, integration problems
- **BILLING**: Invoice issues, payment failures, refunds, subscriptions
- **ACCOUNT**: Login issues, registration, password resets, permissions
- **COMPLAINT**: Customer dissatisfaction, poor service, policy issues
- **REQUEST**: Feature requests, documentation, general questions, feedback

## 🎯 Priority Levels

- **HIGH**: Critical outages, payment blockages, loss of core functionality
- **MEDIUM**: Important issues with workarounds, minor billing issues, access delays
- **LOW**: Non-blocking queries, feature requests, minor styling bugs

## 🤝 Contributing

1. Create a feature branch
2. Implement your changes following the existing code structure
3. Add tests for new functionality
4. Update documentation as needed
5. Open a pull request with a clear description

## 👥 Collaborators

- Mohamed Amine Jguirim
- Fares Tekaya
- Firas Hizem

## 📄 License

This project is part of the Intelligent Support Platform internship project.

## 🔗 Related Projects

- [Backend Service](https://github.com/aminejguirim10/intelligent-support-platform-backend) - Spring Boot REST API
- [Frontend Application](https://github.com/aminejguirim10/intelligent-support-platform-frontend) - Angular web application
