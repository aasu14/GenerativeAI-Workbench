import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.database import get_db
from database.models import Agent, generate_uuid

router = APIRouter()


class GuardrailsSchema(BaseModel):
    max_tokens_per_response: int = 4096
    max_tokens_per_minute: int = 100000
    content_filter_enabled: bool = True
    allowed_domains: list[str] = []
    blocked_keywords: list[str] = []


class InteractionRulesSchema(BaseModel):
    allowed_collaborators: list[str] = []
    escalation_agent_id: Optional[str] = None
    max_turns: int = 20
    auto_summarize: bool = True


class AgentCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    role: str = Field(..., min_length=1, max_length=255)
    description: str = ""
    system_prompt: str = "You are a helpful AI assistant."
    model: str = "gpt-4o-mini"
    tools: list[str] = []
    channels: list[str] = []
    schedule: Optional[str] = None
    memory_enabled: bool = True
    skills: list[str] = []
    guardrails: GuardrailsSchema = GuardrailsSchema()
    interaction_rules: InteractionRulesSchema = InteractionRulesSchema()


class AgentUpdate(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    model: Optional[str] = None
    tools: Optional[list[str]] = None
    channels: Optional[list[str]] = None
    schedule: Optional[str] = None
    memory_enabled: Optional[bool] = None
    skills: Optional[list[str]] = None
    guardrails: Optional[GuardrailsSchema] = None
    interaction_rules: Optional[InteractionRulesSchema] = None


class AgentResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: str
    name: str
    role: str
    description: str
    system_prompt: str
    model: str
    tools: list[str]
    channels: list[str]
    schedule: Optional[str]
    memory_enabled: bool
    skills: list[str]
    guardrails: dict
    interaction_rules: dict
    status: str
    created_at: datetime.datetime
    updated_at: datetime.datetime


@router.get("/", response_model=list[AgentResponse])
async def list_agents(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Agent).order_by(Agent.created_at.desc()))
    agents = result.scalars().all()
    return agents


@router.post("/", response_model=AgentResponse, status_code=201)
async def create_agent(agent_data: AgentCreate, db: AsyncSession = Depends(get_db)):
    agent = Agent(
        id=generate_uuid(),
        name=agent_data.name,
        role=agent_data.role,
        description=agent_data.description,
        system_prompt=agent_data.system_prompt,
        model=agent_data.model,
        tools=agent_data.tools,
        channels=agent_data.channels,
        schedule=agent_data.schedule,
        memory_enabled=agent_data.memory_enabled,
        skills=agent_data.skills,
        guardrails=agent_data.guardrails.model_dump(),
        interaction_rules=agent_data.interaction_rules.model_dump(),
    )
    db.add(agent)
    await db.commit()
    await db.refresh(agent)
    return agent


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(agent_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Agent).where(Agent.id == agent_id))
    agent = result.scalar_one_or_none()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent


@router.put("/{agent_id}", response_model=AgentResponse)
async def update_agent(agent_id: str, agent_data: AgentUpdate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Agent).where(Agent.id == agent_id))
    agent = result.scalar_one_or_none()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    update_data = agent_data.model_dump(exclude_unset=True)
    if "guardrails" in update_data and update_data["guardrails"] is not None:
        update_data["guardrails"] = agent_data.guardrails.model_dump()
    if "interaction_rules" in update_data and update_data["interaction_rules"] is not None:
        update_data["interaction_rules"] = agent_data.interaction_rules.model_dump()

    for field, value in update_data.items():
        setattr(agent, field, value)

    agent.updated_at = datetime.datetime.utcnow()
    await db.commit()
    await db.refresh(agent)
    return agent


@router.delete("/{agent_id}", status_code=204)
async def delete_agent(agent_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Agent).where(Agent.id == agent_id))
    agent = result.scalar_one_or_none()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    await db.delete(agent)
    await db.commit()


@router.post("/{agent_id}/status")
async def update_agent_status(agent_id: str, status: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Agent).where(Agent.id == agent_id))
    agent = result.scalar_one_or_none()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    if status not in ("idle", "running", "paused", "error"):
        raise HTTPException(status_code=400, detail="Invalid status")
    agent.status = status
    await db.commit()
    return {"status": agent.status}
