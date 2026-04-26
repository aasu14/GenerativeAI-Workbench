# Cortex

An AI agent orchestration platform for creating, configuring, and orchestrating collaborative AI agents with visual workflow building, real-time monitoring, and external messaging channel integration.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (React + TypeScript)            │
│  ┌──────────┐ ┌──────────────┐ ┌──────────┐ ┌──────────────┐  │
│  │ Agent    │ │ Workflow     │ │ Live     │ │ Template     │  │
│  │ Builder  │ │ Builder      │ │ Monitor  │ │ Gallery      │  │
│  └──────────┘ └──────────────┘ └──────────┘ └──────────────┘  │
│         React Flow (Visual DAG)  │  WebSocket (Real-time)      │
└───────────────────────┬─────────────────────────────────────────┘
                        │ REST API + WebSocket
┌───────────────────────┴─────────────────────────────────────────┐
│                     Backend (FastAPI + Python)                   │
│  ┌──────────────┐  ┌────────────────┐  ┌────────────────────┐  │
│  │ API Layer    │  │ Agent Runtime  │  │ Channel Manager    │  │
│  │ (CRUD, WS)  │  │ (LangGraph)    │  │ (Telegram Bot)     │  │
│  └──────────────┘  └────────────────┘  └────────────────────┘  │
│  ┌──────────────┐  ┌────────────────┐  ┌────────────────────┐  │
│  │ Persistence  │  │ Message Bus    │  │ Tool Registry      │  │
│  │ (SQLite+SA)  │  │ (AsyncIO Queue)│  │ (Web/Calc/Code)    │  │
│  └──────────────┘  └────────────────┘  └────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Technology Choices & Justification

### Backend: Python + FastAPI
- **Why Python**: First-class support for all major AI frameworks (LangGraph, CrewAI, AutoGen). Richest ecosystem for LLM tooling.
- **Why FastAPI**: Native async support critical for concurrent agent execution. Built-in WebSocket support for real-time monitoring. Auto-generated OpenAPI docs.

### AI Framework: LangGraph
- **Why LangGraph**: Provides stateful, graph-based agent orchestration with explicit control flow. Unlike CrewAI (which is more opinionated), LangGraph allows building custom conditional edges, feedback loops, and human-in-the-loop patterns — all required by the visual workflow builder. State persistence is built-in.

### Frontend: React + TypeScript + Tailwind CSS + React Flow
- **Why React Flow**: Industry-standard library for visual node-based editors. Perfect for the workflow builder with drag-and-drop, conditions, and feedback loops.
- **Why Tailwind**: Rapid UI development with consistent design system.
- **Design Language**: Apple-inspired glassmorphism — dark mesh background, frosted-glass panels (`backdrop-blur`), subtle white-opacity borders, SF Pro typography, and minimalist rounded controls.

### Database: SQLite + SQLAlchemy
- **Why SQLite**: Zero-config, runs fully local, single-file database. Perfect for the "single setup command" requirement.
- **Why SQLAlchemy**: ORM with async support, easy migration to PostgreSQL if needed.

### Messaging: Telegram Bot API
- **Why Telegram**: Free bot creation, rich API, supports text/images/files, easy local development (no business verification like WhatsApp). `python-telegram-bot` library has excellent async support.

## Features

### Agent Management
- Full CRUD for agents with configurable: name, role, system prompt, model, tools, channels
- Schedule configuration (cron-based), memory persistence, skills assignment
- Guardrails (token limits, content filters, rate limiting)
- Interaction rules (allowed collaborators, escalation policies)

### Workflow Builder
- Visual DAG editor with drag-and-drop nodes
- Conditional branching and feedback loops
- 2 pre-built templates: Research & Report, Customer Support Escalation
- Real-time execution visualization

### Monitoring
- Live logs with WebSocket streaming
- Inter-agent message history with full conversation view
- Token usage and cost tracking per agent and per workflow
- Execution timeline visualization

### Telegram Integration
- Any agent can be connected to Telegram for human conversation
- Message history synced to platform UI
- Supports text commands and natural language

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- OpenAI API key (or compatible LLM provider)

### Single Command Setup
```bash
chmod +x setup.sh && ./setup.sh
```

### Manual Setup
```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
uvicorn main:app --reload --port 8000

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

### Docker Setup
```bash
docker-compose up --build
```

### Environment Variables
| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for LLM calls | Yes |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token from @BotFather | For Telegram |
| `DATABASE_URL` | SQLite path (default: sqlite:///./agents.db) | No |

## Running Tests
```bash
cd backend
pytest tests/ -v
```

## Adding New Workflow Templates

1. Create a new template definition in `backend/templates/`:
```python
TEMPLATE = {
    "name": "My Template",
    "description": "What this template does",
    "agents": [...],
    "graph": {"nodes": [...], "edges": [...]}
}
```
2. Register it in `backend/api/templates.py`

## Adding New Messaging Channels

1. Create a new channel class extending `BaseChannel` in `backend/channels/`:
```python
class SlackChannel(BaseChannel):
    async def send_message(self, text: str) -> None: ...
    async def start_listening(self) -> None: ...
```
2. Register the channel in `backend/channels/__init__.py`
3. Add the channel type to the agent configuration schema

## Project Structure
```
ai-agent-platform/
├── backend/
│   ├── main.py              # FastAPI application entry
│   ├── config.py             # Configuration management
│   ├── database/
│   │   ├── models.py         # SQLAlchemy models
│   │   └── database.py       # DB connection & session
│   ├── api/
│   │   ├── agents.py         # Agent CRUD endpoints
│   │   ├── workflows.py      # Workflow management
│   │   ├── monitoring.py     # WebSocket monitoring
│   │   └── templates.py      # Pre-built templates
│   ├── runtime/
│   │   ├── engine.py         # LangGraph orchestration
│   │   ├── executor.py       # Agent execution logic
│   │   └── tools.py          # Tool registry
│   ├── channels/
│   │   ├── base.py           # Base channel interface
│   │   └── telegram_bot.py   # Telegram integration
│   └── tests/
│       ├── test_agents.py
│       ├── test_workflows.py
│       └── test_messages.py
├── frontend/
│   └── src/
│       ├── pages/            # Dashboard, AgentBuilder, etc.
│       ├── components/       # Reusable UI components
│       └── services/         # API client
├── docker-compose.yml
├── setup.sh
└── README.md
```

## License
MIT
