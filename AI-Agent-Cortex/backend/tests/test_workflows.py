import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_create_workflow(client: AsyncClient):
    response = await client.post("/api/workflows/", json={
        "name": "Test Workflow",
        "description": "A test workflow",
        "agents": [],
        "graph": {
            "nodes": [
                {"id": "start", "type": "start", "label": "Start", "position": {"x": 0, "y": 0}, "config": {}},
                {"id": "end", "type": "end", "label": "End", "position": {"x": 0, "y": 200}, "config": {}},
            ],
            "edges": [
                {"id": "e1", "source": "start", "target": "end", "label": "Direct"},
            ],
        },
    })
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Test Workflow"
    assert len(data["graph"]["nodes"]) == 2
    assert len(data["graph"]["edges"]) == 1


@pytest.mark.asyncio
async def test_list_workflows(client: AsyncClient):
    await client.post("/api/workflows/", json={"name": "WF1", "description": ""})
    await client.post("/api/workflows/", json={"name": "WF2", "description": ""})

    response = await client.get("/api/workflows/")
    assert response.status_code == 200
    assert len(response.json()) == 2


@pytest.mark.asyncio
async def test_update_workflow(client: AsyncClient):
    create_resp = await client.post("/api/workflows/", json={"name": "Original", "description": ""})
    wf_id = create_resp.json()["id"]

    response = await client.put(f"/api/workflows/{wf_id}", json={
        "name": "Updated",
        "graph": {
            "nodes": [
                {"id": "start", "type": "start", "label": "Start", "position": {"x": 0, "y": 0}, "config": {}},
            ],
            "edges": [],
        },
    })
    assert response.status_code == 200
    assert response.json()["name"] == "Updated"


@pytest.mark.asyncio
async def test_delete_workflow(client: AsyncClient):
    create_resp = await client.post("/api/workflows/", json={"name": "Delete Me", "description": ""})
    wf_id = create_resp.json()["id"]

    assert (await client.delete(f"/api/workflows/{wf_id}")).status_code == 204
    assert (await client.get(f"/api/workflows/{wf_id}")).status_code == 404


@pytest.mark.asyncio
async def test_workflow_with_agents(client: AsyncClient):
    # Create agents
    a1 = await client.post("/api/agents/", json={"name": "Researcher", "role": "researcher"})
    a2 = await client.post("/api/agents/", json={"name": "Writer", "role": "writer"})
    a1_id = a1.json()["id"]
    a2_id = a2.json()["id"]

    # Create workflow with those agents
    response = await client.post("/api/workflows/", json={
        "name": "Multi-Agent",
        "description": "Two agent workflow",
        "agents": [a1_id, a2_id],
        "graph": {
            "nodes": [
                {"id": "start", "type": "start", "label": "Start", "position": {"x": 250, "y": 0}, "config": {}},
                {"id": "n1", "type": "agent", "label": "Researcher", "position": {"x": 250, "y": 100}, "config": {"agent_id": a1_id, "role": "researcher"}},
                {"id": "n2", "type": "agent", "label": "Writer", "position": {"x": 250, "y": 200}, "config": {"agent_id": a2_id, "role": "writer"}},
                {"id": "end", "type": "end", "label": "End", "position": {"x": 250, "y": 300}, "config": {}},
            ],
            "edges": [
                {"id": "e1", "source": "start", "target": "n1", "label": "Begin"},
                {"id": "e2", "source": "n1", "target": "n2", "label": "Pass findings"},
                {"id": "e3", "source": "n2", "target": "end", "label": "Done"},
            ],
        },
    })
    assert response.status_code == 201
    data = response.json()
    assert len(data["agents"]) == 2
    assert len(data["graph"]["nodes"]) == 4


@pytest.mark.asyncio
async def test_list_templates(client: AsyncClient):
    response = await client.get("/api/templates/")
    assert response.status_code == 200
    templates = response.json()
    assert len(templates) >= 2  # At least 2 built-in templates
    names = [t["name"] for t in templates]
    assert "Research & Report" in names
    assert "Customer Support Escalation" in names


@pytest.mark.asyncio
async def test_instantiate_template(client: AsyncClient):
    response = await client.post("/api/templates/builtin-research-&-report/instantiate")
    assert response.status_code == 200
    data = response.json()
    assert data["workflow_id"]
    assert len(data["agents_created"]) == 3  # researcher, writer, reviewer
