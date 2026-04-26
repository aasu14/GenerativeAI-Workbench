import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Bot, GitBranch, Activity, MessageSquare, Zap, DollarSign,
  ArrowRight, RefreshCw
} from 'lucide-react';
import { monitoringApi, agentsApi, workflowsApi } from '../services/api';
import { MonitoringStats, Agent, Workflow } from '../types';

export default function Dashboard() {
  const [stats, setStats] = useState<MonitoringStats | null>(null);
  const [agents, setAgents] = useState<Agent[]>([]);
  const [workflows, setWorkflows] = useState<Workflow[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchData = async () => {
    try {
      const [s, a, w] = await Promise.all([
        monitoringApi.getStats(),
        agentsApi.list(),
        workflowsApi.list(),
      ]);
      setStats(s);
      setAgents(a);
      setWorkflows(w);
    } catch (err) {
      console.error('Failed to load dashboard:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchData(); }, []);

  const statCards = stats ? [
    { label: 'Agents', value: stats.total_agents, icon: Bot, gradient: 'from-blue-500/20 to-blue-600/5', accent: 'text-blue-400', link: '/agents' },
    { label: 'Active', value: stats.active_agents, icon: Zap, gradient: 'from-green-500/20 to-green-600/5', accent: 'text-green-400', link: '/agents' },
    { label: 'Messages', value: stats.total_messages, icon: MessageSquare, gradient: 'from-purple-500/20 to-purple-600/5', accent: 'text-purple-400', link: '/monitoring' },
    { label: 'Running', value: stats.running_executions, icon: Activity, gradient: 'from-orange-500/20 to-orange-600/5', accent: 'text-orange-400', link: '/workflows' },
    { label: 'Tokens', value: stats.total_tokens.toLocaleString(), icon: Zap, gradient: 'from-cyan-500/20 to-cyan-600/5', accent: 'text-cyan-400', link: '/monitoring' },
    { label: 'Cost', value: `$${stats.total_cost.toFixed(4)}`, icon: DollarSign, gradient: 'from-yellow-500/20 to-yellow-600/5', accent: 'text-yellow-400', link: '/monitoring' },
  ] : [];

  return (
    <div className="p-8 max-w-6xl mx-auto">
      <div className="flex items-center justify-between mb-10">
        <div>
          <h1 className="text-[32px] font-semibold text-white tracking-tight">Dashboard</h1>
          <p className="text-[15px] text-white/40 mt-1">Overview of your AI orchestration platform</p>
        </div>
        <button onClick={fetchData} className="btn-secondary flex items-center gap-2">
          <RefreshCw className="w-4 h-4" /> Refresh
        </button>
      </div>

      {loading ? (
        <div className="flex items-center justify-center h-64">
          <div className="w-6 h-6 border-2 border-white/10 border-t-white/60 rounded-full animate-spin" />
        </div>
      ) : (
        <>
          <div className="grid grid-cols-2 lg:grid-cols-3 gap-4 mb-10">
            {statCards.map(({ label, value, icon: Icon, gradient, accent, link }) => (
              <Link key={label} to={link} className="glass glass-hover rounded-2xl p-5 transition-all duration-300 cursor-pointer">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-[13px] text-white/40 font-medium">{label}</p>
                    <p className="text-[28px] font-semibold text-white mt-1 tracking-tight">{value}</p>
                  </div>
                  <div className={`w-10 h-10 rounded-xl bg-gradient-to-br ${gradient} flex items-center justify-center`}>
                    <Icon className={`w-5 h-5 ${accent}`} strokeWidth={1.8} />
                  </div>
                </div>
              </Link>
            ))}
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="glass rounded-2xl overflow-hidden">
              <div className="flex items-center justify-between px-5 py-4 border-b border-white/[0.06]">
                <h2 className="text-[15px] font-semibold text-white">Recent Agents</h2>
                <Link to="/agents" className="text-[13px] text-blue-400 hover:text-blue-300 flex items-center gap-1">
                  View All <ArrowRight className="w-3.5 h-3.5" />
                </Link>
              </div>
              <div className="p-2">
                {agents.length === 0 ? (
                  <p className="text-white/30 text-[13px] text-center py-8">No agents yet. <Link to="/agents" className="text-blue-400">Create one</Link></p>
                ) : agents.slice(0, 5).map(agent => (
                  <Link key={agent.id} to="/agents" className="flex items-center justify-between px-3 py-2.5 rounded-xl hover:bg-white/[0.04] transition-colors">
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 rounded-lg bg-blue-500/10 flex items-center justify-center">
                        <Bot className="w-4 h-4 text-blue-400" strokeWidth={1.8} />
                      </div>
                      <div>
                        <p className="text-[13px] font-medium text-white">{agent.name}</p>
                        <p className="text-[11px] text-white/30">{agent.role}</p>
                      </div>
                    </div>
                    <span className={`text-[11px] px-2 py-0.5 rounded-full font-medium ${
                      agent.status === 'running' ? 'bg-green-500/15 text-green-400' :
                      agent.status === 'error' ? 'bg-red-500/15 text-red-400' :
                      'bg-white/[0.06] text-white/40'
                    }`}>{agent.status}</span>
                  </Link>
                ))}
              </div>
            </div>

            <div className="glass rounded-2xl overflow-hidden">
              <div className="flex items-center justify-between px-5 py-4 border-b border-white/[0.06]">
                <h2 className="text-[15px] font-semibold text-white">Recent Workflows</h2>
                <Link to="/workflows" className="text-[13px] text-blue-400 hover:text-blue-300 flex items-center gap-1">
                  View All <ArrowRight className="w-3.5 h-3.5" />
                </Link>
              </div>
              <div className="p-2">
                {workflows.length === 0 ? (
                  <p className="text-white/30 text-[13px] text-center py-8">No workflows yet. <Link to="/templates" className="text-blue-400">Start from a template</Link></p>
                ) : workflows.slice(0, 5).map(wf => (
                  <Link key={wf.id} to="/workflows" className="flex items-center justify-between px-3 py-2.5 rounded-xl hover:bg-white/[0.04] transition-colors">
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 rounded-lg bg-purple-500/10 flex items-center justify-center">
                        <GitBranch className="w-4 h-4 text-purple-400" strokeWidth={1.8} />
                      </div>
                      <div>
                        <p className="text-[13px] font-medium text-white">{wf.name}</p>
                        <p className="text-[11px] text-white/30">{wf.agents.length} agents</p>
                      </div>
                    </div>
                    <span className={`text-[11px] px-2 py-0.5 rounded-full font-medium ${
                      wf.status === 'running' ? 'bg-green-500/15 text-green-400' : 'bg-white/[0.06] text-white/40'
                    }`}>{wf.status}</span>
                  </Link>
                ))}
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
