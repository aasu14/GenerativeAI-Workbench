import datetime
import uuid
from sqlalchemy import Column, String, Text, Boolean, DateTime, Float, Integer, ForeignKey, JSON
from sqlalchemy.orm import relationship
from database.database import Base


def generate_uuid():
    return str(uuid.uuid4())


class Agent(Base):
    __tablename__ = "agents"

    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String(255), nullable=False)
    role = Column(String(255), nullable=False)
    description = Column(Text, default="")
    system_prompt = Column(Text, default="You are a helpful AI assistant.")
    model = Column(String(100), default="gpt-4o-mini")
    tools = Column(JSON, default=list)
    channels = Column(JSON, default=list)
    schedule = Column(String(100), nullable=True)
    memory_enabled = Column(Boolean, default=True)
    skills = Column(JSON, default=list)
    guardrails = Column(JSON, default=dict)
    interaction_rules = Column(JSON, default=dict)
    status = Column(String(50), default="idle")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    sent_messages = relationship("Message", foreign_keys="Message.from_agent_id", back_populates="from_agent")
    received_messages = relationship("Message", foreign_keys="Message.to_agent_id", back_populates="to_agent")


class Workflow(Base):
    __tablename__ = "workflows"

    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String(255), nullable=False)
    description = Column(Text, default="")
    agents = Column(JSON, default=list)
    graph = Column(JSON, default=dict)
    is_template = Column(Boolean, default=False)
    status = Column(String(50), default="draft")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    executions = relationship("WorkflowExecution", back_populates="workflow", cascade="all, delete-orphan")


class WorkflowExecution(Base):
    __tablename__ = "workflow_executions"

    id = Column(String, primary_key=True, default=generate_uuid)
    workflow_id = Column(String, ForeignKey("workflows.id"), nullable=False)
    status = Column(String(50), default="pending")
    input_data = Column(JSON, default=dict)
    result = Column(JSON, nullable=True)
    total_tokens = Column(Integer, default=0)
    total_cost = Column(Float, default=0.0)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    workflow = relationship("Workflow", back_populates="executions")
    messages = relationship("Message", back_populates="execution", order_by="Message.created_at", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"

    id = Column(String, primary_key=True, default=generate_uuid)
    execution_id = Column(String, ForeignKey("workflow_executions.id"), nullable=True)
    from_agent_id = Column(String, ForeignKey("agents.id"), nullable=True)
    to_agent_id = Column(String, ForeignKey("agents.id"), nullable=True)
    content = Column(Text, nullable=False)
    message_type = Column(String(50), default="text")
    channel = Column(String(50), default="internal")
    tokens_used = Column(Integer, default=0)
    cost = Column(Float, default=0.0)
    metadata_ = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    execution = relationship("WorkflowExecution", back_populates="messages")
    from_agent = relationship("Agent", foreign_keys=[from_agent_id], back_populates="sent_messages")
    to_agent = relationship("Agent", foreign_keys=[to_agent_id], back_populates="received_messages")


class AgentMemory(Base):
    __tablename__ = "agent_memories"

    id = Column(String, primary_key=True, default=generate_uuid)
    agent_id = Column(String, ForeignKey("agents.id"), nullable=False)
    key = Column(String(255), nullable=False)
    value = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
