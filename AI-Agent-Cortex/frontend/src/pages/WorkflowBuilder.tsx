import { useEffect, useState, useCallback } from 'react';
import ReactFlow, {
  Node, Edge, addEdge, Connection, useNodesState, useEdgesState,
  Controls, Background, MiniMap, Panel, MarkerType, Handle, Position,
  NodeProps,
} from 'reactflow';
import 'reactflow/dist/style.css';
import {
  Plus, Play, Save, Trash2, GitBranch, Bot, CircleDot, Square,
  Diamond, ArrowRight, X, List
} from 'lucide-react';
import { workflowsApi, agentsApi } from '../services/api';
import { Workflow, Agent, WorkflowExecution } from '../types';

// Custom Node Components
function StartNode({ data }: NodeProps) {
  return (
    <div className="bg-green-500/10 border border-green-500/30 backdrop-blur-xl rounded-full px-5 py-2.5 text-green-400 text-[13px] font-medium">
      <Handle type="source" position={Position.Bottom} className="!bg-green-400 !w-2 !h-2" />
      {data.label}
    </div>
  );
}

function EndNode({ data }: NodeProps) {
  return (
    <div className="bg-red-500/10 border border-red-500/30 backdrop-blur-xl rounded-full px-5 py-2.5 text-red-400 text-[13px] font-medium">
      <Handle type="target" position={Position.Top} className="!bg-red-400 !w-2 !h-2" />
      {data.label}
    </div>
  );
}

function AgentNode({ data }: NodeProps) {
  return (
    <div className="bg-white/[0.04] border border-white/[0.1] backdrop-blur-xl rounded-xl px-4 py-3 min-w-[160px]">
      <Handle type="target" position={Position.Top} className="!bg-blue-400 !w-2 !h-2" />
      <div className="flex items-center gap-2">
        <Bot className="w-4 h-4 text-blue-400" strokeWidth={1.8} />
        <span className="text-[13px] font-medium text-white">{data.label}</span>
      </div>
      {data.role && <p className="text-[11px] text-white/30 mt-1">{data.role}</p>}
      <Handle type="source" position={Position.Bottom} className="!bg-blue-400 !w-2 !h-2" />
    </div>
  );
}

function ConditionNode({ data }: NodeProps) {
  return (
    <div className="bg-yellow-500/[0.06] border border-yellow-500/20 backdrop-blur-xl rounded-xl px-4 py-3 min-w-[140px]">
      <Handle type="target" position={Position.Top} className="!bg-yellow-400 !w-2 !h-2" />
      <div className="flex items-center gap-2">
        <Diamond className="w-4 h-4 text-yellow-400" strokeWidth={1.8} />
        <span className="text-[13px] font-medium text-yellow-300">{data.label}</span>
      </div>
      <Handle type="source" position={Position.Bottom} className="!bg-yellow-400 !w-2 !h-2" id="default" />
      <Handle type="source" position={Position.Right} className="!bg-yellow-400 !w-2 !h-2" id="alt" />
    </div>
  );
}

const nodeTypes = {
  start: StartNode,
  end: EndNode,
  agent: AgentNode,
  condition: ConditionNode,
};

export default function WorkflowBuilder() {
  const [workflows, setWorkflows] = useState<Workflow[]>([]);
  const [agents, setAgents] = useState<Agent[]>([]);
  const [selectedWorkflow, setSelectedWorkflow] = useState<Workflow | null>(null);
  const [showList, setShowList] = useState(true);
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [showExecuteModal, setShowExecuteModal] = useState(false);
  const [executeInput, setExecuteInput] = useState('');
  const [executions, setExecutions] = useState<WorkflowExecution[]>([]);
  const [showNewWorkflow, setShowNewWorkflow] = useState(false);
  const [newWorkflowName, setNewWorkflowName] = useState('');
  const [newWorkflowDesc, setNewWorkflowDesc] = useState('');
  const [showAddNode, setShowAddNode] = useState(false);
  const [executionResult, setExecutionResult] = useState<WorkflowExecution | null>(null);
  const [showResults, setShowResults] = useState(false);
  const [pollingExec, setPollingExec] = useState(false);

  useEffect(() => {
    Promise.all([workflowsApi.list(), agentsApi.list()])
      .then(([w, a]) => { setWorkflows(w); setAgents(a); })
      .catch(console.error);
  }, []);

  const onConnect = useCallback(
    (params: Connection) => setEdges(eds => addEdge({
      ...params,
      markerEnd: { type: MarkerType.ArrowClosed },
      style: { stroke: '#6b7280' },
      label: '',
    }, eds)),
    [setEdges]
  );

  const loadWorkflow = async (wf: Workflow) => {
    setSelectedWorkflow(wf);
    setShowList(false);

    const graph = wf.graph;
    const flowNodes: Node[] = (graph.nodes || []).map((n: any) => ({
      id: n.id,
      type: n.type,
      position: n.position || { x: 0, y: 0 },
      data: {
        label: n.label,
        role: n.config?.role,
        ...n.config,
      },
    }));

    const flowEdges: Edge[] = (graph.edges || []).map((e: any) => ({
      id: e.id,
      source: e.source,
      target: e.target,
      label: e.label || '',
      markerEnd: { type: MarkerType.ArrowClosed },
      style: { stroke: e.condition ? '#eab308' : '#6b7280' },
      animated: !!e.condition,
    }));

    setNodes(flowNodes);
    setEdges(flowEdges);

    // Load executions
    try {
      const execs = await workflowsApi.listExecutions(wf.id);
      setExecutions(execs);
    } catch {
      setExecutions([]);
    }
  };

  const handleSave = async () => {
    if (!selectedWorkflow) return;

    const graph = {
      nodes: nodes.map(n => ({
        id: n.id,
        type: n.type || 'agent',
        label: n.data.label,
        position: n.position,
        config: { role: n.data.role, agent_id: n.data.agent_id },
      })),
      edges: edges.map(e => ({
        id: e.id,
        source: e.source,
        target: e.target,
        label: typeof e.label === 'string' ? e.label : '',
        condition: e.animated ? 'condition' : undefined,
      })),
    };

    try {
      const updated = await workflowsApi.update(selectedWorkflow.id, { graph });
      setSelectedWorkflow(updated);
      const wfs = await workflowsApi.list();
      setWorkflows(wfs);
    } catch (err) {
      alert(`Save error: ${err}`);
    }
  };

  const handleExecute = async () => {
    if (!selectedWorkflow) return;
    try {
      const execution = await workflowsApi.execute(selectedWorkflow.id, {
        query: executeInput || 'Execute the workflow',
      });
      setShowExecuteModal(false);
      setExecuteInput('');
      setExecutions(prev => [execution, ...prev]);
      setExecutionResult(execution);
      setShowResults(true);
      setPollingExec(true);

      // Poll for completion
      const poll = setInterval(async () => {
        try {
          const updated = await workflowsApi.getExecution(execution.id);
          setExecutionResult(updated);
          if (updated.status === 'completed' || updated.status === 'failed') {
            clearInterval(poll);
            setPollingExec(false);
            // Refresh executions list
            const execs = await workflowsApi.listExecutions(selectedWorkflow.id);
            setExecutions(execs);
          }
        } catch {
          clearInterval(poll);
          setPollingExec(false);
        }
      }, 2000);
    } catch (err) {
      alert(`Execution error: ${err}`);
    }
  };

  const handleCreateWorkflow = async () => {
    if (!newWorkflowName) return;
    try {
      const wf = await workflowsApi.create({
        name: newWorkflowName,
        description: newWorkflowDesc,
        agents: [],
        graph: {
          nodes: [
            { id: 'start', type: 'start', label: 'Start', position: { x: 250, y: 0 }, config: {} },
            { id: 'end', type: 'end', label: 'End', position: { x: 250, y: 400 }, config: {} },
          ],
          edges: [],
        },
      });
      setWorkflows(prev => [wf, ...prev]);
      setShowNewWorkflow(false);
      setNewWorkflowName('');
      setNewWorkflowDesc('');
      loadWorkflow(wf);
    } catch (err) {
      alert(`Error: ${err}`);
    }
  };

  const addNode = (type: string, agentId?: string) => {
    const agent = agentId ? agents.find(a => a.id === agentId) : null;
    const id = `node_${Date.now()}`;
    const newNode: Node = {
      id,
      type,
      position: { x: 250, y: 200 + nodes.length * 100 },
      data: {
        label: type === 'condition' ? 'Condition' : agent ? agent.name : type,
        role: agent?.role,
        agent_id: agentId,
      },
    };
    setNodes(prev => [...prev, newNode]);
    setShowAddNode(false);
  };

  const deleteSelected = () => {
    setNodes(nds => nds.filter(n => !n.selected));
    setEdges(eds => eds.filter(e => !e.selected));
  };

  return (
    <div className="h-full flex">
      {/* Workflow List Sidebar */}
      {showList && (
        <div className="w-72 bg-black/40 backdrop-blur-xl border-r border-white/[0.06] flex flex-col">
          <div className="p-4 border-b border-white/[0.06]">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-[15px] font-semibold text-white">Workflows</h2>
              <button onClick={() => setShowNewWorkflow(true)}
                className="p-1.5 bg-blue-500 hover:bg-blue-600 rounded-lg text-white transition-colors">
                <Plus className="w-4 h-4" />
              </button>
            </div>

            {showNewWorkflow && (
              <div className="space-y-2 mb-3">
                <input value={newWorkflowName} onChange={e => setNewWorkflowName(e.target.value)}
                  className="input-glass w-full text-[13px]" placeholder="Workflow name" />
                <input value={newWorkflowDesc} onChange={e => setNewWorkflowDesc(e.target.value)}
                  className="input-glass w-full text-[13px]" placeholder="Description" />
                <div className="flex gap-2">
                  <button onClick={handleCreateWorkflow} className="btn-primary flex-1 text-[13px] py-1.5">Create</button>
                  <button onClick={() => setShowNewWorkflow(false)} className="btn-secondary text-[13px] py-1.5">Cancel</button>
                </div>
              </div>
            )}
          </div>

          <div className="flex-1 overflow-y-auto p-4 space-y-2">
            {workflows.length === 0 ? (
              <p className="text-[13px] text-white/30 text-center py-8">No workflows yet</p>
            ) : (
              workflows.map(wf => (
                <div
                  key={wf.id}
                  className={`relative group w-full text-left p-3 rounded-xl transition-all duration-200 cursor-pointer ${
                    selectedWorkflow?.id === wf.id
                      ? 'bg-white/[0.08] border border-white/[0.12]'
                      : 'hover:bg-white/[0.04] border border-transparent'
                  }`}
                  onClick={() => loadWorkflow(wf)}
                >
                  <div className="flex items-center gap-2">
                    <GitBranch className="w-4 h-4 text-purple-400" strokeWidth={1.8} />
                    <span className="text-[13px] font-medium text-white flex-1">{wf.name}</span>
                    <button
                      onClick={async (e) => {
                        e.stopPropagation();
                        if (!confirm(`Delete workflow "${wf.name}"?`)) return;
                        try {
                          await workflowsApi.delete(wf.id);
                          setWorkflows(prev => prev.filter(w => w.id !== wf.id));
                          if (selectedWorkflow?.id === wf.id) {
                            setSelectedWorkflow(null);
                            setNodes([]);
                            setEdges([]);
                          }
                        } catch (err) {
                          alert(`Delete failed: ${err}`);
                        }
                      }}
                      className="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-500/10 rounded-md text-white/20 hover:text-red-400 transition-all"
                      title="Delete workflow"
                    >
                      <Trash2 className="w-3 h-3" />
                    </button>
                  </div>
                  <p className="text-[11px] text-white/30 mt-1 line-clamp-2">{wf.description}</p>
                  <p className="text-[11px] text-white/20 mt-1">{wf.agents.length} agents</p>
                </div>
              ))
            )}
          </div>
        </div>
      )}

      {/* Workflow Canvas */}
      <div className="flex-1 relative">
        {selectedWorkflow ? (
          <>
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onConnect={onConnect}
              nodeTypes={nodeTypes}
              fitView
              className="!bg-transparent"
            >
              <Background color="rgba(255,255,255,0.04)" gap={24} />
              <Controls className="!bg-white/[0.04] !border-white/[0.08] !rounded-xl !backdrop-blur-xl" />
              <MiniMap
                className="!bg-white/[0.04] !border-white/[0.08] !rounded-xl !backdrop-blur-xl"
                nodeColor={n =>
                  n.type === 'start' ? '#22c55e' :
                  n.type === 'end' ? '#ef4444' :
                  n.type === 'condition' ? '#eab308' :
                  '#3b82f6'
                }
              />

              <Panel position="top-left" className="flex gap-2">
                <button onClick={() => setShowList(!showList)}
                  className="glass glass-hover p-2 rounded-xl text-white/50 hover:text-white transition-colors">
                  <List className="w-4 h-4" />
                </button>
              </Panel>

              <Panel position="top-right" className="flex gap-2">
                <button onClick={() => setShowAddNode(true)}
                  className="btn-primary flex items-center gap-1.5 text-[13px] py-1.5">
                  <Plus className="w-4 h-4" /> Add Node
                </button>
                <button onClick={deleteSelected}
                  className="glass glass-hover flex items-center gap-1.5 px-3 py-1.5 rounded-full text-white/50 text-[13px]">
                  <Trash2 className="w-4 h-4" />
                </button>
                <button onClick={handleSave}
                  className="glass glass-hover flex items-center gap-1.5 px-3 py-1.5 rounded-full text-white/50 text-[13px]">
                  <Save className="w-4 h-4" /> Save
                </button>
                <button onClick={() => setShowExecuteModal(true)}
                  className="flex items-center gap-1.5 px-4 py-1.5 bg-green-500 hover:bg-green-600 rounded-full text-white text-[13px] font-medium transition-colors">
                  <Play className="w-4 h-4" /> Execute
                </button>
              </Panel>
            </ReactFlow>

            {/* Add Node Modal */}
            {showAddNode && (
              <div className="absolute inset-0 bg-black/40 backdrop-blur-sm flex items-center justify-center z-50">
                <div className="glass rounded-2xl p-6 w-96 max-h-[80vh] overflow-y-auto border border-white/[0.08]">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-[15px] font-semibold text-white">Add Node</h3>
                    <button onClick={() => setShowAddNode(false)} className="text-white/30 hover:text-white transition-colors">
                      <X className="w-5 h-5" />
                    </button>
                  </div>

                  <div className="space-y-2 mb-4">
                    <p className="text-[12px] font-medium text-white/40 uppercase tracking-wider">Control</p>
                    <button onClick={() => addNode('condition')} className="w-full text-left p-3 rounded-xl hover:bg-white/[0.04] transition-colors">
                      <div className="flex items-center gap-2">
                        <Diamond className="w-4 h-4 text-yellow-400" strokeWidth={1.8} />
                        <span className="text-[13px] text-white">Condition</span>
                      </div>
                      <p className="text-[11px] text-white/30 mt-1">Add conditional branching</p>
                    </button>
                  </div>

                  <div className="space-y-2">
                    <p className="text-[12px] font-medium text-white/40 uppercase tracking-wider">Agents</p>
                    {agents.length === 0 ? (
                      <p className="text-[13px] text-white/30">No agents available.</p>
                    ) : agents.map(agent => (
                      <button key={agent.id} onClick={() => addNode('agent', agent.id)}
                        className="w-full text-left p-3 rounded-xl hover:bg-white/[0.04] transition-colors">
                        <div className="flex items-center gap-2">
                          <Bot className="w-4 h-4 text-blue-400" strokeWidth={1.8} />
                          <span className="text-[13px] text-white">{agent.name}</span>
                          <span className="text-[11px] text-white/30">({agent.role})</span>
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Execute Modal */}
            {showExecuteModal && (
              <div className="absolute inset-0 bg-black/40 backdrop-blur-sm flex items-center justify-center z-50">
                <div className="glass rounded-2xl p-6 w-96 border border-white/[0.08]">
                  <h3 className="text-[15px] font-semibold text-white mb-4">Execute Workflow</h3>
                  <div className="mb-4">
                    <label className="block text-[13px] text-white/40 mb-1.5">Input / Query</label>
                    <textarea value={executeInput} onChange={e => setExecuteInput(e.target.value)}
                      rows={4} className="input-glass w-full resize-none text-[13px]"
                      placeholder="What should the workflow accomplish?" />
                  </div>
                  <div className="flex gap-2">
                    <button onClick={handleExecute}
                      className="flex-1 flex items-center justify-center gap-2 px-4 py-2 bg-green-500 hover:bg-green-600 rounded-full text-white text-[13px] font-medium transition-colors">
                      <Play className="w-4 h-4" /> Execute
                    </button>
                    <button onClick={() => setShowExecuteModal(false)} className="btn-secondary">Cancel</button>
                  </div>

                  {/* Recent executions */}
                  {executions.length > 0 && (
                    <div className="mt-4 border-t border-white/[0.06] pt-4">
                      <p className="text-[12px] font-medium text-white/40 mb-2">Recent</p>
                      <div className="space-y-2 max-h-40 overflow-y-auto">
                        {executions.slice(0, 5).map(exec => (
                          <div key={exec.id} className="p-2 bg-white/[0.03] rounded-lg text-[11px]">
                            <div className="flex items-center justify-between">
                              <span className={`px-1.5 py-0.5 rounded-md ${
                                exec.status === 'completed' ? 'bg-green-500/15 text-green-400' :
                                exec.status === 'running' ? 'bg-blue-500/15 text-blue-400' :
                                exec.status === 'failed' ? 'bg-red-500/15 text-red-400' :
                                'bg-white/[0.06] text-white/40'
                              }`}>
                                {exec.status}
                              </span>
                              <span className="text-white/25">
                                {exec.total_tokens} tokens · ${exec.total_cost.toFixed(4)}
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Execution Results Panel */}
            {showResults && executionResult && (
              <div className="absolute bottom-0 left-0 right-0 bg-black/60 backdrop-blur-xl border-t border-white/[0.08] z-40 max-h-[50vh] overflow-y-auto">
                <div className="p-4">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-3">
                      <h3 className="text-[15px] font-semibold text-white">Results</h3>
                      <span className={`text-[11px] px-2 py-0.5 rounded-full font-medium ${
                        executionResult.status === 'completed' ? 'bg-green-500/15 text-green-400' :
                        executionResult.status === 'running' ? 'bg-blue-500/15 text-blue-400 animate-pulse' :
                        executionResult.status === 'failed' ? 'bg-red-500/15 text-red-400' :
                        'bg-white/[0.06] text-white/40'
                      }`}>
                        {executionResult.status === 'running' ? '⏳ Running...' : executionResult.status}
                      </span>
                      {executionResult.total_tokens > 0 && (
                        <span className="text-[11px] text-white/30">
                          {executionResult.total_tokens} tokens · ${executionResult.total_cost.toFixed(4)}
                        </span>
                      )}
                    </div>
                    <button onClick={() => setShowResults(false)} className="text-white/30 hover:text-white transition-colors">
                      <X className="w-5 h-5" />
                    </button>
                  </div>

                  {pollingExec && (
                    <div className="flex items-center gap-2 text-blue-400 text-[13px] mb-3">
                      <div className="w-4 h-4 border-2 border-blue-400/20 border-t-blue-400 rounded-full animate-spin" />
                      Agents working...
                    </div>
                  )}

                  {executionResult.result && (
                    <div className="space-y-3">
                      {/* Agent messages */}
                      {executionResult.result.messages?.map((msg: any, i: number) => (
                        <div key={i} className="glass rounded-xl p-3">
                          <div className="flex items-center gap-2 mb-2">
                            <Bot className="w-4 h-4 text-blue-400" strokeWidth={1.8} />
                            <span className="text-[13px] font-medium text-blue-400">{msg.agent}</span>
                          </div>
                          <p className="text-[13px] text-white/70 whitespace-pre-wrap">{msg.content}</p>
                        </div>
                      ))}

                      {/* Final output */}
                      {executionResult.result.output && !executionResult.result.messages?.length && (
                        <div className="glass rounded-xl p-3">
                          <p className="text-[13px] text-white/70 whitespace-pre-wrap">{executionResult.result.output}</p>
                        </div>
                      )}

                      {/* Error */}
                      {executionResult.result.error && (
                        <div className="glass rounded-xl p-3 border-red-500/20">
                          <p className="text-[13px] text-red-400">{executionResult.result.error}</p>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            )}
          </>
        ) : (
          <div className="h-full flex items-center justify-center">
            <div className="text-center">
              <GitBranch className="w-12 h-12 text-white/10 mx-auto mb-4" strokeWidth={1.2} />
              <h3 className="text-[17px] font-medium text-white/50 mb-2">No Workflow Selected</h3>
              <p className="text-[13px] text-white/30 mb-5">Select a workflow or create a new one</p>
              <button
                onClick={() => { setShowList(true); setShowNewWorkflow(true); }}
                className="btn-primary"
              >
                Create Workflow
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
