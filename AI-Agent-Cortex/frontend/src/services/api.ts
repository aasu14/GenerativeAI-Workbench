import { Agent, Workflow, WorkflowExecution, Message, MonitoringStats, Template } from '../types';

const API_BASE = '/api';

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${url}`, {
    headers: { 'Content-Type': 'application/json', ...options?.headers },
    ...options,
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Request failed' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }
  if (response.status === 204) return undefined as T;
  return response.json();
}

// Agents
export const agentsApi = {
  list: () => request<Agent[]>('/agents/'),
  get: (id: string) => request<Agent>(`/agents/${id}`),
  create: (data: Partial<Agent>) => request<Agent>('/agents/', { method: 'POST', body: JSON.stringify(data) }),
  update: (id: string, data: Partial<Agent>) => request<Agent>(`/agents/${id}`, { method: 'PUT', body: JSON.stringify(data) }),
  delete: (id: string) => request<void>(`/agents/${id}`, { method: 'DELETE' }),
};

// Workflows
export const workflowsApi = {
  list: () => request<Workflow[]>('/workflows/'),
  get: (id: string) => request<Workflow>(`/workflows/${id}`),
  create: (data: Partial<Workflow>) => request<Workflow>('/workflows/', { method: 'POST', body: JSON.stringify(data) }),
  update: (id: string, data: Partial<Workflow>) => request<Workflow>(`/workflows/${id}`, { method: 'PUT', body: JSON.stringify(data) }),
  delete: (id: string) => request<void>(`/workflows/${id}`, { method: 'DELETE' }),
  execute: (id: string, input_data: Record<string, any> = {}) =>
    request<WorkflowExecution>(`/workflows/${id}/execute`, { method: 'POST', body: JSON.stringify({ input_data }) }),
  listExecutions: (id: string) => request<WorkflowExecution[]>(`/workflows/${id}/executions`),
  getExecution: (id: string) => request<WorkflowExecution>(`/workflows/executions/${id}`),
  getExecutionMessages: (id: string) => request<Message[]>(`/workflows/executions/${id}/messages`),
};

// Monitoring
export const monitoringApi = {
  getStats: () => request<MonitoringStats>('/monitoring/stats'),
  getMessages: (limit = 50) => request<Message[]>(`/monitoring/messages?limit=${limit}`),
};

// Templates
export const templatesApi = {
  list: () => request<Template[]>('/templates/'),
  instantiate: (id: string) =>
    request<{ workflow_id: string; agents_created: string[]; message: string }>(`/templates/${id}/instantiate`, { method: 'POST' }),
};

// WebSocket
export function createMonitoringSocket(onMessage: (data: any) => void): WebSocket {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const ws = new WebSocket(`${protocol}//${window.location.hostname}:8000/api/monitoring/ws`);

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onMessage(data);
    } catch (e) {
      console.error('WS parse error:', e);
    }
  };

  ws.onclose = () => {
    // Auto-reconnect after 3 seconds
    setTimeout(() => {
      createMonitoringSocket(onMessage);
    }, 3000);
  };

  // Ping every 30s to keep alive
  const pingInterval = setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'ping' }));
    }
  }, 30000);

  ws.addEventListener('close', () => clearInterval(pingInterval));

  return ws;
}
