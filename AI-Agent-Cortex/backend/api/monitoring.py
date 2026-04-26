import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from database.database import get_db, async_session
from database.models import Message, WorkflowExecution, Agent

router = APIRouter()
logger = logging.getLogger(__name__)

# Global list of connected WebSocket clients
connected_clients: list[WebSocket] = []


class MonitoringStats(BaseModel):
    total_agents: int
    active_agents: int
    total_workflows: int
    running_executions: int
    total_messages: int
    total_tokens: int
    total_cost: float


async def broadcast_event(event_type: str, data: dict):
    """Broadcast an event to all connected WebSocket clients."""
    message = json.dumps({"type": event_type, "data": data})
    disconnected = []
    for client in connected_clients:
        try:
            await client.send_text(message)
        except Exception:
            disconnected.append(client)
    for client in disconnected:
        connected_clients.remove(client)


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    logger.info(f"WebSocket client connected. Total: {len(connected_clients)}")

    try:
        while True:
            data = await websocket.receive_text()
            # Handle ping/pong or commands from client
            try:
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        connected_clients.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total: {len(connected_clients)}")


@router.get("/stats", response_model=MonitoringStats)
async def get_stats(db: AsyncSession = Depends(get_db)):
    total_agents = (await db.execute(select(func.count(Agent.id)))).scalar() or 0
    active_agents = (await db.execute(
        select(func.count(Agent.id)).where(Agent.status == "running")
    )).scalar() or 0
    total_workflows = (await db.execute(
        select(func.count(WorkflowExecution.id))
    )).scalar() or 0
    running_executions = (await db.execute(
        select(func.count(WorkflowExecution.id)).where(WorkflowExecution.status == "running")
    )).scalar() or 0
    total_messages = (await db.execute(select(func.count(Message.id)))).scalar() or 0
    total_tokens = (await db.execute(select(func.coalesce(func.sum(Message.tokens_used), 0)))).scalar() or 0
    total_cost = (await db.execute(select(func.coalesce(func.sum(Message.cost), 0)))).scalar() or 0.0

    return MonitoringStats(
        total_agents=total_agents,
        active_agents=active_agents,
        total_workflows=total_workflows,
        running_executions=running_executions,
        total_messages=total_messages,
        total_tokens=total_tokens,
        total_cost=total_cost,
    )


@router.get("/messages")
async def get_recent_messages(limit: int = 50, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Message).order_by(Message.created_at.desc()).limit(min(limit, 200))
    )
    messages = result.scalars().all()

    # Resolve agent names
    agent_ids = set()
    for m in messages:
        if m.from_agent_id:
            agent_ids.add(m.from_agent_id)
        if m.to_agent_id:
            agent_ids.add(m.to_agent_id)

    agent_names: dict[str, str] = {}
    if agent_ids:
        agents_result = await db.execute(
            select(Agent).where(Agent.id.in_(agent_ids))
        )
        for agent in agents_result.scalars().all():
            agent_names[agent.id] = agent.name

    return [
        {
            "id": m.id,
            "execution_id": m.execution_id,
            "from_agent_id": m.from_agent_id,
            "from_agent_name": agent_names.get(m.from_agent_id, None) if m.from_agent_id else None,
            "to_agent_id": m.to_agent_id,
            "to_agent_name": agent_names.get(m.to_agent_id, None) if m.to_agent_id else None,
            "content": m.content,
            "message_type": m.message_type,
            "channel": m.channel,
            "tokens_used": m.tokens_used,
            "cost": m.cost,
            "created_at": m.created_at.isoformat(),
        }
        for m in messages
    ]
