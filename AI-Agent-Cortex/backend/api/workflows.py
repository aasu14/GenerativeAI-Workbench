import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.database import get_db
from database.models import Workflow, WorkflowExecution, Message, generate_uuid

router = APIRouter()


class WorkflowNodeSchema(BaseModel):
    id: str
    type: str  # "agent", "condition", "start", "end"
    agent_id: Optional[str] = None
    label: str = ""
    position: dict = {"x": 0, "y": 0}
    config: dict = {}


class WorkflowEdgeSchema(BaseModel):
    id: str
    source: str
    target: str
    label: str = ""
    condition: Optional[str] = None


class WorkflowGraphSchema(BaseModel):
    nodes: list[WorkflowNodeSchema] = []
    edges: list[WorkflowEdgeSchema] = []


class WorkflowCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str = ""
    agents: list[str] = []
    graph: WorkflowGraphSchema = WorkflowGraphSchema()


class WorkflowUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    agents: Optional[list[str]] = None
    graph: Optional[WorkflowGraphSchema] = None


class WorkflowResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: str
    name: str
    description: str
    agents: list
    graph: dict
    is_template: bool
    status: str
    created_at: datetime.datetime
    updated_at: datetime.datetime


class ExecutionCreate(BaseModel):
    input_data: dict = {}


class ExecutionResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: str
    workflow_id: str
    status: str
    input_data: dict
    result: Optional[dict]
    total_tokens: int
    total_cost: float
    started_at: Optional[datetime.datetime]
    completed_at: Optional[datetime.datetime]
    created_at: datetime.datetime


class MessageResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: str
    execution_id: Optional[str]
    from_agent_id: Optional[str]
    to_agent_id: Optional[str]
    content: str
    message_type: str
    channel: str
    tokens_used: int
    cost: float
    created_at: datetime.datetime


@router.get("/", response_model=list[WorkflowResponse])
async def list_workflows(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Workflow).where(Workflow.is_template == False).order_by(Workflow.created_at.desc())
    )
    return result.scalars().all()


@router.post("/", response_model=WorkflowResponse, status_code=201)
async def create_workflow(data: WorkflowCreate, db: AsyncSession = Depends(get_db)):
    workflow = Workflow(
        id=generate_uuid(),
        name=data.name,
        description=data.description,
        agents=data.agents,
        graph=data.graph.model_dump(),
    )
    db.add(workflow)
    await db.commit()
    await db.refresh(workflow)
    return workflow


@router.get("/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(workflow_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Workflow).where(Workflow.id == workflow_id))
    workflow = result.scalar_one_or_none()
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return workflow


@router.put("/{workflow_id}", response_model=WorkflowResponse)
async def update_workflow(workflow_id: str, data: WorkflowUpdate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Workflow).where(Workflow.id == workflow_id))
    workflow = result.scalar_one_or_none()
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")

    update_data = data.model_dump(exclude_unset=True)
    if "graph" in update_data and update_data["graph"] is not None:
        update_data["graph"] = data.graph.model_dump()

    for field, value in update_data.items():
        setattr(workflow, field, value)

    workflow.updated_at = datetime.datetime.utcnow()
    await db.commit()
    await db.refresh(workflow)
    return workflow


@router.delete("/{workflow_id}", status_code=204)
async def delete_workflow(workflow_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Workflow).where(Workflow.id == workflow_id))
    workflow = result.scalar_one_or_none()
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    await db.delete(workflow)
    await db.commit()


@router.post("/{workflow_id}/execute", response_model=ExecutionResponse)
async def execute_workflow(workflow_id: str, data: ExecutionCreate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Workflow).where(Workflow.id == workflow_id))
    workflow = result.scalar_one_or_none()
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")

    execution = WorkflowExecution(
        id=generate_uuid(),
        workflow_id=workflow_id,
        status="running",
        input_data=data.input_data,
        started_at=datetime.datetime.utcnow(),
    )
    db.add(execution)
    await db.commit()
    await db.refresh(execution)

    # Launch execution asynchronously
    import asyncio
    from runtime.engine import WorkflowEngine
    engine = WorkflowEngine(db)
    asyncio.create_task(engine.run(execution.id, workflow, data.input_data))

    return execution


@router.get("/{workflow_id}/executions", response_model=list[ExecutionResponse])
async def list_executions(workflow_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(WorkflowExecution)
        .where(WorkflowExecution.workflow_id == workflow_id)
        .order_by(WorkflowExecution.created_at.desc())
    )
    return result.scalars().all()


@router.get("/executions/{execution_id}", response_model=ExecutionResponse)
async def get_execution(execution_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(WorkflowExecution).where(WorkflowExecution.id == execution_id)
    )
    execution = result.scalar_one_or_none()
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")
    return execution


@router.get("/executions/{execution_id}/messages", response_model=list[MessageResponse])
async def get_execution_messages(execution_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Message)
        .where(Message.execution_id == execution_id)
        .order_by(Message.created_at)
    )
    return result.scalars().all()
