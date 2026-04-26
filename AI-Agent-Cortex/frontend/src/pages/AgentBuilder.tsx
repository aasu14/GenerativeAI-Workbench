import { useEffect, useState } from 'react';
import { Bot, Plus, Trash2, Edit, Save, X, Settings, Shield, Users } from 'lucide-react';
import { agentsApi } from '../services/api';
import { Agent } from '../types';

const AVAILABLE_TOOLS = ['web_search', 'calculator', 'text_analysis', 'current_time', 'code_executor'];
const AVAILABLE_CHANNELS = ['telegram', 'internal'];
const AVAILABLE_MODELS = ['gpt-4o-mini', 'gpt-4o', 'gpt-4-turbo', 'gpt-3.5-turbo'];

interface AgentForm {
  name: string;
  role: string;
  description: string;
  system_prompt: string;
  model: string;
  tools: string[];
  channels: string[];
  schedule: string;
  memory_enabled: boolean;
  skills: string[];
  guardrails: {
    max_tokens_per_response: number;
    max_tokens_per_minute: number;
    content_filter_enabled: boolean;
    allowed_domains: string[];
    blocked_keywords: string[];
  };
  interaction_rules: {
    allowed_collaborators: string[];
    escalation_agent_id: string | null;
    max_turns: number;
    auto_summarize: boolean;
  };
}

const defaultAgent: AgentForm = {
  name: '',
  role: '',
  description: '',
  system_prompt: 'You are a helpful AI assistant.',
  model: 'gpt-4o-mini',
  tools: [],
  channels: [],
  schedule: '',
  memory_enabled: true,
  skills: [],
  guardrails: {
    max_tokens_per_response: 4096,
    max_tokens_per_minute: 100000,
    content_filter_enabled: true,
    allowed_domains: [],
    blocked_keywords: [],
  },
  interaction_rules: {
    allowed_collaborators: [],
    escalation_agent_id: null,
    max_turns: 20,
    auto_summarize: true,
  },
};

export default function AgentBuilder() {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [showForm, setShowForm] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [form, setForm] = useState<AgentForm>({ ...defaultAgent });
  const [activeTab, setActiveTab] = useState<'basic' | 'tools' | 'guardrails' | 'interaction'>('basic');
  const [loading, setLoading] = useState(true);

  const fetchAgents = async () => {
    try {
      const data = await agentsApi.list();
      setAgents(data);
    } catch (err) {
      console.error('Failed to load agents:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchAgents(); }, []);

  const handleSave = async () => {
    try {
      if (editingId) {
        await agentsApi.update(editingId, form as any);
      } else {
        await agentsApi.create(form as any);
      }
      setShowForm(false);
      setEditingId(null);
      setForm({ ...defaultAgent });
      fetchAgents();
    } catch (err) {
      alert(`Error: ${err}`);
    }
  };

  const handleEdit = (agent: Agent) => {
    setForm({
      name: agent.name,
      role: agent.role,
      description: agent.description,
      system_prompt: agent.system_prompt,
      model: agent.model,
      tools: agent.tools,
      channels: agent.channels,
      schedule: agent.schedule || '',
      memory_enabled: agent.memory_enabled,
      skills: agent.skills,
      guardrails: agent.guardrails,
      interaction_rules: agent.interaction_rules,
    });
    setEditingId(agent.id);
    setShowForm(true);
    setActiveTab('basic');
  };

  const handleDelete = async (id: string) => {
    if (!confirm('Delete this agent?')) return;
    try {
      await agentsApi.delete(id);
      fetchAgents();
    } catch (err) {
      alert(`Error: ${err}`);
    }
  };

  const toggleArrayItem = (field: 'tools' | 'channels', item: string) => {
    setForm(prev => ({
      ...prev,
      [field]: prev[field].includes(item)
        ? prev[field].filter(i => i !== item)
        : [...prev[field], item],
    }));
  };

  return (
    <div className="p-8 max-w-6xl mx-auto">
      <div className="flex items-center justify-between mb-10">
        <div>
          <h1 className="text-[32px] font-semibold text-white tracking-tight">Agents</h1>
          <p className="text-[15px] text-white/40 mt-1">Create and configure AI agents</p>
        </div>
        <button
          onClick={() => { setShowForm(true); setEditingId(null); setForm({ ...defaultAgent }); setActiveTab('basic'); }}
          className="btn-primary flex items-center gap-2"
        >
          <Plus className="w-4 h-4" /> New Agent
        </button>
      </div>

      {/* Agent Form Modal */}
      {showForm && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <div className="glass rounded-2xl w-full max-w-3xl max-h-[90vh] overflow-y-auto border border-white/[0.08]">
            <div className="flex items-center justify-between px-6 py-5 border-b border-white/[0.06]">
              <h2 className="text-[17px] font-semibold text-white">
                {editingId ? 'Edit Agent' : 'New Agent'}
              </h2>
              <button onClick={() => setShowForm(false)} className="text-white/30 hover:text-white transition-colors">
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="flex border-b border-white/[0.06]">
              {[
                { key: 'basic', label: 'Basic', icon: Bot },
                { key: 'tools', label: 'Tools', icon: Settings },
                { key: 'guardrails', label: 'Guardrails', icon: Shield },
                { key: 'interaction', label: 'Interaction', icon: Users },
              ].map(({ key, label, icon: Icon }) => (
                <button
                  key={key}
                  onClick={() => setActiveTab(key as any)}
                  className={`flex items-center gap-2 px-5 py-3 text-[13px] font-medium border-b-2 transition-all ${
                    activeTab === key
                      ? 'border-blue-500 text-white'
                      : 'border-transparent text-white/40 hover:text-white/70'
                  }`}
                >
                  <Icon className="w-4 h-4" strokeWidth={1.8} /> {label}
                </button>
              ))}
            </div>

            <div className="p-6 space-y-4">
              {activeTab === 'basic' && (
                <>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-[13px] font-medium text-white/60 mb-1.5">Name *</label>
                      <input value={form.name} onChange={e => setForm(p => ({ ...p, name: e.target.value }))}
                        className="input-glass w-full" placeholder="Research Agent" />
                    </div>
                    <div>
                      <label className="block text-[13px] font-medium text-white/60 mb-1.5">Role *</label>
                      <input value={form.role} onChange={e => setForm(p => ({ ...p, role: e.target.value }))}
                        className="input-glass w-full" placeholder="researcher" />
                    </div>
                  </div>
                  <div>
                    <label className="block text-[13px] font-medium text-white/60 mb-1.5">Description</label>
                    <input value={form.description} onChange={e => setForm(p => ({ ...p, description: e.target.value }))}
                      className="input-glass w-full" placeholder="Searches the web for information..." />
                  </div>
                  <div>
                    <label className="block text-[13px] font-medium text-white/60 mb-1.5">System Prompt</label>
                    <textarea value={form.system_prompt} onChange={e => setForm(p => ({ ...p, system_prompt: e.target.value }))}
                      rows={4} className="input-glass w-full resize-none" />
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-[13px] font-medium text-white/60 mb-1.5">Model</label>
                      <select value={form.model} onChange={e => setForm(p => ({ ...p, model: e.target.value }))}
                        className="input-glass w-full">
                        {AVAILABLE_MODELS.map(m => <option key={m} value={m}>{m}</option>)}
                      </select>
                    </div>
                    <div>
                      <label className="block text-[13px] font-medium text-white/60 mb-1.5">Schedule (cron)</label>
                      <input value={form.schedule} onChange={e => setForm(p => ({ ...p, schedule: e.target.value }))}
                        className="input-glass w-full" placeholder="*/30 * * * *" />
                    </div>
                  </div>
                  <label className="flex items-center gap-3 cursor-pointer">
                    <input type="checkbox" checked={form.memory_enabled}
                      onChange={e => setForm(p => ({ ...p, memory_enabled: e.target.checked }))}
                      className="w-4 h-4 rounded border-white/20 bg-white/5 text-blue-500 focus:ring-blue-500/30" />
                    <span className="text-[13px] text-white/60">Enable memory persistence</span>
                  </label>
                </>
              )}

              {activeTab === 'tools' && (
                <>
                  <div>
                    <label className="block text-[13px] font-medium text-white/60 mb-3">Tools</label>
                    <div className="grid grid-cols-2 gap-2">
                      {AVAILABLE_TOOLS.map(tool => (
                        <button key={tool} onClick={() => toggleArrayItem('tools', tool)}
                          className={`flex items-center gap-2 px-4 py-2.5 rounded-xl text-[13px] transition-all ${
                            form.tools.includes(tool)
                              ? 'bg-blue-500/15 text-blue-400 border border-blue-500/30'
                              : 'bg-white/[0.03] text-white/40 border border-white/[0.06] hover:bg-white/[0.06]'
                          }`}>
                          <Settings className="w-3.5 h-3.5" strokeWidth={1.8} /> {tool.replace('_', ' ')}
                        </button>
                      ))}
                    </div>
                  </div>
                  <div>
                    <label className="block text-[13px] font-medium text-white/60 mb-3">Channels</label>
                    <div className="grid grid-cols-2 gap-2">
                      {AVAILABLE_CHANNELS.map(ch => (
                        <button key={ch} onClick={() => toggleArrayItem('channels', ch)}
                          className={`flex items-center gap-2 px-4 py-2.5 rounded-xl text-[13px] transition-all ${
                            form.channels.includes(ch)
                              ? 'bg-green-500/15 text-green-400 border border-green-500/30'
                              : 'bg-white/[0.03] text-white/40 border border-white/[0.06] hover:bg-white/[0.06]'
                          }`}>
                          {ch}
                        </button>
                      ))}
                    </div>
                  </div>
                </>
              )}

              {activeTab === 'guardrails' && (
                <>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-[13px] font-medium text-white/60 mb-1.5">Max Tokens / Response</label>
                      <input type="number" value={form.guardrails.max_tokens_per_response}
                        onChange={e => setForm(p => ({ ...p, guardrails: { ...p.guardrails, max_tokens_per_response: parseInt(e.target.value) || 0 } }))}
                        className="input-glass w-full" />
                    </div>
                    <div>
                      <label className="block text-[13px] font-medium text-white/60 mb-1.5">Max Tokens / Minute</label>
                      <input type="number" value={form.guardrails.max_tokens_per_minute}
                        onChange={e => setForm(p => ({ ...p, guardrails: { ...p.guardrails, max_tokens_per_minute: parseInt(e.target.value) || 0 } }))}
                        className="input-glass w-full" />
                    </div>
                  </div>
                  <label className="flex items-center gap-3 cursor-pointer">
                    <input type="checkbox" checked={form.guardrails.content_filter_enabled}
                      onChange={e => setForm(p => ({ ...p, guardrails: { ...p.guardrails, content_filter_enabled: e.target.checked } }))}
                      className="w-4 h-4 rounded border-white/20 bg-white/5 text-blue-500" />
                    <span className="text-[13px] text-white/60">Content filter</span>
                  </label>
                  <div>
                    <label className="block text-[13px] font-medium text-white/60 mb-1.5">Blocked Keywords</label>
                    <input value={form.guardrails.blocked_keywords.join(', ')}
                      onChange={e => setForm(p => ({ ...p, guardrails: { ...p.guardrails, blocked_keywords: e.target.value.split(',').map(s => s.trim()).filter(Boolean) } }))}
                      className="input-glass w-full" placeholder="keyword1, keyword2" />
                  </div>
                </>
              )}

              {activeTab === 'interaction' && (
                <>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-[13px] font-medium text-white/60 mb-1.5">Max Turns</label>
                      <input type="number" value={form.interaction_rules.max_turns}
                        onChange={e => setForm(p => ({ ...p, interaction_rules: { ...p.interaction_rules, max_turns: parseInt(e.target.value) || 20 } }))}
                        className="input-glass w-full" />
                    </div>
                    <div>
                      <label className="block text-[13px] font-medium text-white/60 mb-1.5">Escalation Agent</label>
                      <select value={form.interaction_rules.escalation_agent_id || ''}
                        onChange={e => setForm(p => ({ ...p, interaction_rules: { ...p.interaction_rules, escalation_agent_id: e.target.value || null } }))}
                        className="input-glass w-full">
                        <option value="">None</option>
                        {agents.filter(a => a.id !== editingId).map(a => (
                          <option key={a.id} value={a.id}>{a.name}</option>
                        ))}
                      </select>
                    </div>
                  </div>
                  <label className="flex items-center gap-3 cursor-pointer">
                    <input type="checkbox" checked={form.interaction_rules.auto_summarize}
                      onChange={e => setForm(p => ({ ...p, interaction_rules: { ...p.interaction_rules, auto_summarize: e.target.checked } }))}
                      className="w-4 h-4 rounded border-white/20 bg-white/5 text-blue-500" />
                    <span className="text-[13px] text-white/60">Auto-summarize conversations</span>
                  </label>
                  <div>
                    <label className="block text-[13px] font-medium text-white/60 mb-2">Collaborators</label>
                    {agents.filter(a => a.id !== editingId).length === 0 ? (
                      <p className="text-[13px] text-white/30">No other agents available</p>
                    ) : agents.filter(a => a.id !== editingId).map(a => (
                      <label key={a.id} className="flex items-center gap-2 text-[13px] text-white/50 mb-1.5 cursor-pointer">
                        <input type="checkbox"
                          checked={form.interaction_rules.allowed_collaborators.includes(a.id)}
                          onChange={e => {
                            const list = e.target.checked
                              ? [...form.interaction_rules.allowed_collaborators, a.id]
                              : form.interaction_rules.allowed_collaborators.filter(id => id !== a.id);
                            setForm(p => ({ ...p, interaction_rules: { ...p.interaction_rules, allowed_collaborators: list } }));
                          }}
                          className="w-4 h-4 rounded border-white/20 bg-white/5" />
                        {a.name} <span className="text-white/30">({a.role})</span>
                      </label>
                    ))}
                  </div>
                </>
              )}
            </div>

            <div className="flex justify-end gap-3 px-6 py-4 border-t border-white/[0.06]">
              <button onClick={() => setShowForm(false)} className="btn-secondary">Cancel</button>
              <button onClick={handleSave} disabled={!form.name || !form.role}
                className="btn-primary flex items-center gap-2 disabled:opacity-40 disabled:cursor-not-allowed">
                <Save className="w-4 h-4" /> {editingId ? 'Update' : 'Create'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Agent List */}
      {loading ? (
        <div className="flex items-center justify-center h-64">
          <div className="w-6 h-6 border-2 border-white/10 border-t-white/60 rounded-full animate-spin" />
        </div>
      ) : agents.length === 0 ? (
        <div className="text-center py-24">
          <Bot className="w-12 h-12 text-white/10 mx-auto mb-4" strokeWidth={1.2} />
          <h3 className="text-[17px] font-medium text-white/50 mb-2">No agents yet</h3>
          <p className="text-[13px] text-white/30 mb-6">Create your first AI agent to get started</p>
          <button onClick={() => { setShowForm(true); setActiveTab('basic'); }} className="btn-primary">
            Create Agent
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {agents.map(agent => (
            <div key={agent.id} className="glass glass-hover rounded-2xl p-5 transition-all duration-300">
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center gap-3">
                  <div className="w-9 h-9 rounded-xl bg-blue-500/10 flex items-center justify-center">
                    <Bot className="w-4.5 h-4.5 text-blue-400" strokeWidth={1.8} />
                  </div>
                  <div>
                    <h3 className="text-[14px] font-semibold text-white">{agent.name}</h3>
                    <p className="text-[12px] text-white/40">{agent.role}</p>
                  </div>
                </div>
                <span className={`text-[11px] px-2 py-0.5 rounded-full font-medium ${
                  agent.status === 'running' ? 'bg-green-500/15 text-green-400' :
                  agent.status === 'error' ? 'bg-red-500/15 text-red-400' :
                  'bg-white/[0.06] text-white/40'
                }`}>{agent.status}</span>
              </div>

              {agent.description && (
                <p className="text-[12px] text-white/35 mb-3 line-clamp-2">{agent.description}</p>
              )}

              <div className="flex flex-wrap gap-1.5 mb-3">
                {agent.tools.map(t => (
                  <span key={t} className="text-[11px] px-2 py-0.5 bg-white/[0.04] text-white/40 rounded-lg border border-white/[0.06]">{t}</span>
                ))}
              </div>

              <div className="text-[11px] text-white/25 mb-4">
                {agent.model} · Memory {agent.memory_enabled ? 'on' : 'off'}
              </div>

              <div className="flex gap-2">
                <button onClick={() => handleEdit(agent)}
                  className="flex items-center gap-1.5 px-3 py-1.5 bg-white/[0.04] hover:bg-white/[0.08] border border-white/[0.06] rounded-lg text-[12px] text-white/60 transition-colors">
                  <Edit className="w-3 h-3" /> Edit
                </button>
                <button onClick={() => handleDelete(agent.id)}
                  className="flex items-center gap-1.5 px-3 py-1.5 bg-red-500/5 hover:bg-red-500/10 border border-red-500/10 rounded-lg text-[12px] text-red-400/70 transition-colors">
                  <Trash2 className="w-3 h-3" /> Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
