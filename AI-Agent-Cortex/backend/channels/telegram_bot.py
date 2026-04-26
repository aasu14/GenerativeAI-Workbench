"""Telegram Bot integration for agent communication."""
import logging

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from sqlalchemy import select

from channels.base import BaseChannel
from config import settings
from database.database import async_session
from database.models import Agent, Message, generate_uuid
from runtime.executor import AgentExecutor

logger = logging.getLogger(__name__)


class TelegramChannel(BaseChannel):
    """Telegram bot that connects to agents configured with 'telegram' channel."""

    def __init__(self):
        self.app: Application = None
        self.agent_executors: dict[str, AgentExecutor] = {}
        self._running = False

    async def _get_telegram_agents(self) -> list[dict]:
        """Fetch all agents configured for Telegram."""
        async with async_session() as db:
            result = await db.execute(select(Agent))
            agents = result.scalars().all()
            telegram_agents = []
            for agent in agents:
                channels = agent.channels or []
                if "telegram" in channels:
                    telegram_agents.append({
                        "id": agent.id,
                        "name": agent.name,
                        "role": agent.role,
                        "system_prompt": agent.system_prompt,
                        "model": agent.model,
                        "tools": agent.tools or [],
                        "guardrails": agent.guardrails or {},
                    })
            return telegram_agents

    async def _ensure_executors(self):
        """Initialize or refresh agent executors."""
        agents = await self._get_telegram_agents()
        for agent_config in agents:
            if agent_config["id"] not in self.agent_executors:
                self.agent_executors[agent_config["id"]] = AgentExecutor(agent_config)

    async def _start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        await self._ensure_executors()
        if not self.agent_executors:
            await update.message.reply_text(
                "👋 Welcome to the AI Agent Platform!\n\n"
                "No agents are currently connected to Telegram. "
                "Please configure an agent with the 'telegram' channel in the web UI."
            )
            return

        agent_list = []
        for agent_id, executor in self.agent_executors.items():
            agent_list.append(f"• **{executor.name}** ({executor.role})")

        await update.message.reply_text(
            f"👋 Welcome to the AI Agent Platform!\n\n"
            f"Available agents:\n" + "\n".join(agent_list) + "\n\n"
            f"Use /agent <name> to switch agents, or just send a message to chat with the default agent.\n"
            f"Use /agents to list available agents.\n"
            f"Use /clear to reset conversation history.",
            parse_mode="Markdown",
        )

    async def _agents_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /agents command."""
        await self._ensure_executors()
        if not self.agent_executors:
            await update.message.reply_text("No agents configured for Telegram.")
            return

        lines = ["**Available Agents:**\n"]
        for agent_id, executor in self.agent_executors.items():
            lines.append(f"• **{executor.name}** - {executor.role}")
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    async def _agent_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /agent <name> command to switch active agent."""
        if not context.args:
            await update.message.reply_text("Usage: /agent <agent_name>")
            return

        name = " ".join(context.args).lower()
        await self._ensure_executors()

        for agent_id, executor in self.agent_executors.items():
            if executor.name.lower() == name or executor.role.lower() == name:
                context.user_data["active_agent_id"] = agent_id
                await update.message.reply_text(
                    f"✅ Switched to **{executor.name}** ({executor.role})",
                    parse_mode="Markdown",
                )
                return

        await update.message.reply_text(f"Agent '{name}' not found. Use /agents to see available agents.")

    async def _clear_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /clear command to reset conversation."""
        active_id = context.user_data.get("active_agent_id")
        if active_id and active_id in self.agent_executors:
            self.agent_executors[active_id].memory.clear()
        await update.message.reply_text("🧹 Conversation history cleared.")

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming text messages."""
        await self._ensure_executors()

        if not self.agent_executors:
            await update.message.reply_text(
                "No agents configured for Telegram. Add an agent with 'telegram' channel in the web UI."
            )
            return

        # Get active agent or use first available
        active_id = context.user_data.get("active_agent_id")
        if not active_id or active_id not in self.agent_executors:
            active_id = next(iter(self.agent_executors))
            context.user_data["active_agent_id"] = active_id

        executor = self.agent_executors[active_id]
        user_text = update.message.text

        # Show typing indicator
        await update.message.chat.send_action("typing")

        # Log incoming message
        await self._log_message(
            from_agent_id=None,
            to_agent_id=active_id,
            content=user_text,
            channel="telegram",
            message_type="user_input",
            metadata={"chat_id": str(update.message.chat_id), "username": update.message.from_user.username},
        )

        # Execute agent
        result = await executor.execute(user_text)

        # Log agent response
        await self._log_message(
            from_agent_id=active_id,
            to_agent_id=None,
            content=result["content"],
            channel="telegram",
            message_type="agent_response",
            tokens=result["tokens_used"],
            cost=result["cost"],
            metadata={"chat_id": str(update.message.chat_id)},
        )

        # Broadcast to monitoring WebSocket
        try:
            from api.monitoring import broadcast_event
            await broadcast_event("telegram_message", {
                "agent_id": active_id,
                "agent_name": executor.name,
                "user_message": user_text,
                "agent_response": result["content"][:500],
                "tokens_used": result["tokens_used"],
                "cost": result["cost"],
            })
        except Exception:
            pass

        # Send response (split if too long)
        response_text = result["content"]
        if len(response_text) > 4096:
            for i in range(0, len(response_text), 4096):
                await update.message.reply_text(response_text[i:i + 4096])
        else:
            await update.message.reply_text(response_text)

    async def _log_message(
        self, from_agent_id, to_agent_id, content, channel, message_type,
        tokens=0, cost=0.0, metadata=None,
    ):
        """Persist message to database."""
        try:
            async with async_session() as db:
                msg = Message(
                    id=generate_uuid(),
                    from_agent_id=from_agent_id,
                    to_agent_id=to_agent_id,
                    content=content,
                    channel=channel,
                    message_type=message_type,
                    tokens_used=tokens,
                    cost=cost,
                    metadata_=metadata or {},
                )
                db.add(msg)
                await db.commit()
        except Exception as e:
            logger.error(f"Failed to log telegram message: {e}")

    async def send_message(self, chat_id: str, text: str) -> None:
        """Send a message to a Telegram chat."""
        if self.app and self.app.bot:
            await self.app.bot.send_message(chat_id=int(chat_id), text=text)

    async def start(self) -> None:
        """Start the Telegram bot."""
        if not settings.telegram_bot_token:
            logger.warning("No Telegram bot token configured")
            return

        self.app = Application.builder().token(settings.telegram_bot_token).build()

        # Register handlers
        self.app.add_handler(CommandHandler("start", self._start_command))
        self.app.add_handler(CommandHandler("agents", self._agents_command))
        self.app.add_handler(CommandHandler("agent", self._agent_command))
        self.app.add_handler(CommandHandler("clear", self._clear_command))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message))

        self._running = True
        logger.info("Starting Telegram bot polling...")

        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling(drop_pending_updates=True)

    async def stop(self) -> None:
        """Stop the Telegram bot."""
        if self.app and self._running:
            self._running = False
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()
            logger.info("Telegram bot stopped")

    async def handle_incoming(self, message: dict) -> str:
        """Handle incoming message (used by other parts of the system)."""
        # This is handled by the telegram-bot handlers directly
        return ""
