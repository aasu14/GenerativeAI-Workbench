import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_create_agent(client: AsyncClient):
    response = await client.post("/api/agents/", json={
        "name": "Test Agent",
        "role": "tester",
        "description": "A test agent",
        "system_prompt": "You are a test agent.",
        "model": "gpt-4o-mini",
        "tools": ["web_search", "calculator"],
        "channels": ["telegram"],
        "memory_enabled": True,
    })
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Test Agent"
    assert data["role"] == "tester"
    assert data["tools"] == ["web_search", "calculator"]
    assert data["channels"] == ["telegram"]
    assert data["status"] == "idle"
    assert data["id"]


@pytest.mark.asyncio
async def test_list_agents(client: AsyncClient):
    # Create two agents
    await client.post("/api/agents/", json={"name": "Agent 1", "role": "role1"})
    await client.post("/api/agents/", json={"name": "Agent 2", "role": "role2"})

    response = await client.get("/api/agents/")
    assert response.status_code == 200
    agents = response.json()
    assert len(agents) == 2


@pytest.mark.asyncio
async def test_get_agent(client: AsyncClient):
    create_resp = await client.post("/api/agents/", json={"name": "Get Test", "role": "getter"})
    agent_id = create_resp.json()["id"]

    response = await client.get(f"/api/agents/{agent_id}")
    assert response.status_code == 200
    assert response.json()["name"] == "Get Test"


@pytest.mark.asyncio
async def test_get_agent_not_found(client: AsyncClient):
    response = await client.get("/api/agents/nonexistent-id")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_update_agent(client: AsyncClient):
    create_resp = await client.post("/api/agents/", json={"name": "Before", "role": "updater"})
    agent_id = create_resp.json()["id"]

    response = await client.put(f"/api/agents/{agent_id}", json={
        "name": "After",
        "tools": ["text_analysis"],
    })
    assert response.status_code == 200
    assert response.json()["name"] == "After"
    assert response.json()["tools"] == ["text_analysis"]


@pytest.mark.asyncio
async def test_delete_agent(client: AsyncClient):
    create_resp = await client.post("/api/agents/", json={"name": "Delete Me", "role": "deleter"})
    agent_id = create_resp.json()["id"]

    delete_resp = await client.delete(f"/api/agents/{agent_id}")
    assert delete_resp.status_code == 204

    get_resp = await client.get(f"/api/agents/{agent_id}")
    assert get_resp.status_code == 404


@pytest.mark.asyncio
async def test_agent_guardrails(client: AsyncClient):
    response = await client.post("/api/agents/", json={
        "name": "Guarded Agent",
        "role": "guarded",
        "guardrails": {
            "max_tokens_per_response": 1000,
            "max_tokens_per_minute": 50000,
            "content_filter_enabled": True,
            "allowed_domains": ["example.com"],
            "blocked_keywords": ["secret", "password"],
        },
    })
    assert response.status_code == 201
    data = response.json()
    assert data["guardrails"]["max_tokens_per_response"] == 1000
    assert data["guardrails"]["blocked_keywords"] == ["secret", "password"]


@pytest.mark.asyncio
async def test_agent_interaction_rules(client: AsyncClient):
    # Create collaborator agent first
    collab = await client.post("/api/agents/", json={"name": "Collaborator", "role": "helper"})
    collab_id = collab.json()["id"]

    response = await client.post("/api/agents/", json={
        "name": "Interactive Agent",
        "role": "interactive",
        "interaction_rules": {
            "allowed_collaborators": [collab_id],
            "escalation_agent_id": collab_id,
            "max_turns": 10,
            "auto_summarize": True,
        },
    })
    assert response.status_code == 201
    data = response.json()
    assert collab_id in data["interaction_rules"]["allowed_collaborators"]
    assert data["interaction_rules"]["max_turns"] == 10
