from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.database import get_db
from database.models import Workflow, generate_uuid

router = APIRouter()

# Pre-built workflow templates
TEMPLATES = [
    {
        "name": "Research & Report",
        "description": "A researcher agent gathers information using web search, then a writer agent composes a structured report. Includes quality review feedback loop.",
        "agents": [
            {
                "role": "researcher",
                "name": "Research Agent",
                "system_prompt": "You are a thorough research assistant. Use the web_search tool to find accurate, up-to-date information. Compile your findings into clear bullet points with sources.",
                "tools": ["web_search"],
                "model": "gpt-4o-mini",
            },
            {
                "role": "writer",
                "name": "Report Writer",
                "system_prompt": "You are an expert report writer. Take research findings and compose a well-structured report with an executive summary, key findings, and recommendations.",
                "tools": ["text_analysis"],
                "model": "gpt-4o-mini",
            },
            {
                "role": "reviewer",
                "name": "Quality Reviewer",
                "system_prompt": "You are a critical reviewer. Evaluate the report for accuracy, completeness, and clarity. If issues are found, send feedback back to the writer. If the report is good, approve it.",
                "tools": ["text_analysis"],
                "model": "gpt-4o-mini",
            },
        ],
        "graph": {
            "nodes": [
                {"id": "start", "type": "start", "label": "Start", "position": {"x": 250, "y": 0}, "config": {}},
                {"id": "researcher", "type": "agent", "label": "Research Agent", "position": {"x": 250, "y": 100}, "config": {"role": "researcher"}},
                {"id": "writer", "type": "agent", "label": "Report Writer", "position": {"x": 250, "y": 250}, "config": {"role": "writer"}},
                {"id": "reviewer", "type": "agent", "label": "Quality Reviewer", "position": {"x": 250, "y": 400}, "config": {"role": "reviewer"}},
                {"id": "quality_check", "type": "condition", "label": "Quality OK?", "position": {"x": 250, "y": 550}, "config": {"condition": "approved"}},
                {"id": "end", "type": "end", "label": "End", "position": {"x": 250, "y": 700}, "config": {}},
            ],
            "edges": [
                {"id": "e1", "source": "start", "target": "researcher", "label": "Begin Research"},
                {"id": "e2", "source": "researcher", "target": "writer", "label": "Send Findings"},
                {"id": "e3", "source": "writer", "target": "reviewer", "label": "Submit Report"},
                {"id": "e4", "source": "reviewer", "target": "quality_check", "label": "Review Complete"},
                {"id": "e5", "source": "quality_check", "target": "end", "label": "Approved", "condition": "approved"},
                {"id": "e6", "source": "quality_check", "target": "writer", "label": "Needs Revision", "condition": "revision_needed"},
            ],
        },
    },
    {
        "name": "Customer Support Escalation",
        "description": "A frontline support agent handles initial queries. Complex issues are escalated to a specialist agent. Includes sentiment analysis and auto-categorization.",
        "agents": [
            {
                "role": "frontline",
                "name": "Frontline Support",
                "system_prompt": "You are a friendly customer support agent. Answer simple questions directly. For complex technical issues or billing disputes, escalate to the specialist with a summary of the issue.",
                "tools": ["text_analysis", "calculator"],
                "model": "gpt-4o-mini",
            },
            {
                "role": "specialist",
                "name": "Specialist Agent",
                "system_prompt": "You are a senior technical specialist. Handle escalated issues with detailed analysis. Provide thorough solutions and follow-up recommendations.",
                "tools": ["web_search", "text_analysis", "calculator"],
                "model": "gpt-4o-mini",
            },
            {
                "role": "summarizer",
                "name": "Resolution Summarizer",
                "system_prompt": "You summarize the support interaction into a concise resolution record including: issue category, resolution steps taken, outcome, and customer satisfaction assessment.",
                "tools": ["text_analysis"],
                "model": "gpt-4o-mini",
            },
        ],
        "graph": {
            "nodes": [
                {"id": "start", "type": "start", "label": "Customer Query", "position": {"x": 250, "y": 0}, "config": {}},
                {"id": "frontline", "type": "agent", "label": "Frontline Support", "position": {"x": 250, "y": 100}, "config": {"role": "frontline"}},
                {"id": "complexity_check", "type": "condition", "label": "Needs Escalation?", "position": {"x": 250, "y": 250}, "config": {"condition": "escalate"}},
                {"id": "specialist", "type": "agent", "label": "Specialist Agent", "position": {"x": 450, "y": 350}, "config": {"role": "specialist"}},
                {"id": "summarizer", "type": "agent", "label": "Resolution Summarizer", "position": {"x": 250, "y": 500}, "config": {"role": "summarizer"}},
                {"id": "end", "type": "end", "label": "Resolved", "position": {"x": 250, "y": 650}, "config": {}},
            ],
            "edges": [
                {"id": "e1", "source": "start", "target": "frontline", "label": "Receive Query"},
                {"id": "e2", "source": "frontline", "target": "complexity_check", "label": "Assess"},
                {"id": "e3", "source": "complexity_check", "target": "summarizer", "label": "Simple - Resolved", "condition": "resolved"},
                {"id": "e4", "source": "complexity_check", "target": "specialist", "label": "Complex - Escalate", "condition": "escalate"},
                {"id": "e5", "source": "specialist", "target": "summarizer", "label": "Resolution"},
                {"id": "e6", "source": "summarizer", "target": "end", "label": "Complete"},
            ],
        },
    },
]


@router.get("/")
async def list_templates(db: AsyncSession = Depends(get_db)):
    """List all available workflow templates."""
    # Get DB templates
    result = await db.execute(
        select(Workflow).where(Workflow.is_template == True).order_by(Workflow.created_at.desc())
    )
    db_templates = result.scalars().all()

    templates = []
    for t in TEMPLATES:
        slug = t['name'].lower().replace('&', 'and').replace(' ', '-').replace('--', '-')
        templates.append({
            "id": f"builtin-{slug}",
            "name": t["name"],
            "description": t["description"],
            "agents": t["agents"],
            "graph": t["graph"],
            "is_builtin": True,
        })

    for t in db_templates:
        templates.append({
            "id": t.id,
            "name": t.name,
            "description": t.description,
            "agents": t.agents,
            "graph": t.graph,
            "is_builtin": False,
        })

    return templates


@router.post("/{template_id}/instantiate")
async def instantiate_template(template_id: str, db: AsyncSession = Depends(get_db)):
    """Create a new workflow from a template."""
    template_data = None

    # Check built-in templates
    for t in TEMPLATES:
        slug = t['name'].lower().replace('&', 'and').replace(' ', '-').replace('--', '-')
        if f"builtin-{slug}" == template_id:
            template_data = t
            break

    # Check DB templates
    if not template_data:
        result = await db.execute(select(Workflow).where(Workflow.id == template_id, Workflow.is_template == True))
        db_template = result.scalar_one_or_none()
        if db_template:
            template_data = {
                "name": db_template.name,
                "description": db_template.description,
                "agents": db_template.agents,
                "graph": db_template.graph,
            }

    if not template_data:
        raise HTTPException(status_code=404, detail="Template not found")

    # Create agents from template (reuse existing agents with matching name+role)
    from database.models import Agent
    agent_id_map = {}
    created_agents = []

    for agent_def in template_data.get("agents", []):
        # Check if agent with same name and role already exists
        existing = await db.execute(
            select(Agent).where(Agent.name == agent_def["name"], Agent.role == agent_def["role"])
        )
        existing_agent = existing.scalar_one_or_none()

        if existing_agent:
            agent_id_map[agent_def["role"]] = existing_agent.id
            created_agents.append(existing_agent.id)
        else:
            agent = Agent(
                id=generate_uuid(),
                name=agent_def["name"],
                role=agent_def["role"],
                system_prompt=agent_def["system_prompt"],
                tools=agent_def.get("tools", []),
                model=agent_def.get("model", "gpt-4o-mini"),
            )
            db.add(agent)
            agent_id_map[agent_def["role"]] = agent.id
            created_agents.append(agent.id)

    # Create workflow with mapped agent IDs
    graph = template_data["graph"].copy()
    for node in graph.get("nodes", []):
        if node.get("config", {}).get("role") in agent_id_map:
            node["config"]["agent_id"] = agent_id_map[node["config"]["role"]]

    workflow = Workflow(
        id=generate_uuid(),
        name=f"{template_data['name']} (from template)",
        description=template_data["description"],
        agents=created_agents,
        graph=graph,
        is_template=False,
    )
    db.add(workflow)
    await db.commit()
    await db.refresh(workflow)

    return {
        "workflow_id": workflow.id,
        "agents_created": created_agents,
        "message": f"Workflow '{workflow.name}' created from template",
    }
