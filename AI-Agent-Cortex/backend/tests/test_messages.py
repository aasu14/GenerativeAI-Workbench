import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_monitoring_stats(client: AsyncClient):
    response = await client.get("/api/monitoring/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_agents" in data
    assert "active_agents" in data
    assert "total_messages" in data
    assert "total_tokens" in data
    assert "total_cost" in data


@pytest.mark.asyncio
async def test_monitoring_messages_empty(client: AsyncClient):
    response = await client.get("/api/monitoring/messages")
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_monitoring_stats_after_agent_creation(client: AsyncClient):
    await client.post("/api/agents/", json={"name": "Stats Agent", "role": "stat"})

    response = await client.get("/api/monitoring/stats")
    data = response.json()
    assert data["total_agents"] == 1
    assert data["active_agents"] == 0


@pytest.mark.asyncio
async def test_health_check(client: AsyncClient):
    response = await client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "1.0.0"
