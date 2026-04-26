export interface Agent {
  id: string;
  name: string;
  role: string;
  description: string;
  system_prompt: string;
  model: string;
  tools: string[];
  channels: string[];
  schedule: string | null;
  memory_enabled: boolean;
  skills: string[];
  guardrails: Guardrails;
  interaction_rules: InteractionRules;
  status: string;
  created_at: string;
  updated_at: string;
}

export interface Guardrails {
  max_tokens_per_response: number;
  max_tokens_per_minute: number;
  content_filter_enabled: boolean;
  allowed_domains: string[];
  blocked_keywords: string[];
}

export interface InteractionRules {
  allowed_collaborators: string[];
  escalation_agent_id: string | null;
  max_turns: number;
  auto_summarize: boolean;
}

export interface WorkflowNode {
  id: string;
  type: string;
  label: string;
  position: { x: number; y: number };
  config: Record<string, any>;
}

export interface WorkflowEdge {
  id: string;
  source: string;
  target: string;
  label: string;
  condition?: string;
}

export interface WorkflowGraph {
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
}

export interface Workflow {
  id: string;
  name: string;
  description: string;
  agents: string[];
  graph: WorkflowGraph;
  is_template: boolean;
  status: string;
  created_at: string;
  updated_at: string;
}

export interface WorkflowExecution {
  id: string;
  workflow_id: string;
  status: string;
  input_data: Record<string, any>;
  result: Record<string, any> | null;
  total_tokens: number;
  total_cost: number;
  started_at: string | null;
  completed_at: string | null;
  created_at: string;
}

export interface Message {
  id: string;
  execution_id: string | null;
  from_agent_id: string | null;
  to_agent_id: string | null;
  content: string;
  message_type: string;
  channel: string;
  tokens_used: number;
  cost: number;
  created_at: string;
}

export interface MonitoringStats {
  total_agents: number;
  active_agents: number;
  total_workflows: number;
  running_executions: number;
  total_messages: number;
  total_tokens: number;
  total_cost: number;
}

export interface Template {
  id: string;
  name: string;
  description: string;
  agents: any[];
  graph: WorkflowGraph;
  is_builtin: boolean;
}
