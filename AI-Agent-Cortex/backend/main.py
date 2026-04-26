import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from database.database import init_db
from api.agents import router as agents_router
from api.workflows import router as workflows_router
from api.monitoring import router as monitoring_router
from api.templates import router as templates_router

logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AI Agent Orchestration Platform...")
    await init_db()

    # Start Telegram bot if token configured
    if settings.telegram_bot_token and settings.telegram_bot_token != "your-telegram-bot-token":
        from channels.telegram_bot import TelegramChannel
        telegram = TelegramChannel()
        asyncio.create_task(telegram.start())
        app.state.telegram = telegram
        logger.info("Telegram bot started")

    yield

    # Shutdown
    if hasattr(app.state, "telegram"):
        await app.state.telegram.stop()
    logger.info("Platform shutdown complete")


app = FastAPI(
    title="AI Agent Orchestration Platform",
    description="Create, configure, and orchestrate collaborative AI agents",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(agents_router, prefix="/api/agents", tags=["agents"])
app.include_router(workflows_router, prefix="/api/workflows", tags=["workflows"])
app.include_router(monitoring_router, prefix="/api/monitoring", tags=["monitoring"])
app.include_router(templates_router, prefix="/api/templates", tags=["templates"])


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}
