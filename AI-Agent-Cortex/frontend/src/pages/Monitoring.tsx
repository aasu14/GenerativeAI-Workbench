import { useEffect, useState, useCallback } from 'react';
import {
  Activity, MessageSquare, Zap, DollarSign, RefreshCw, Wifi, WifiOff
} from 'lucide-react';
import { monitoringApi, createMonitoringSocket } from '../services/api';
import { MonitoringStats, Message } from '../types';

export default function Monitoring() {
  const [stats, setStats] = useState<MonitoringStats | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [liveEvents, setLiveEvents] = useState<any[]>([]);
  const [wsConnected, setWsConnected] = useState(false);

  const fetchData = async () => {
    try {
      const [s, m] = await Promise.all([
        monitoringApi.getStats(),
        monitoringApi.getMessages(100),
      ]);
      setStats(s);
      setMessages(m);
    } catch (err) {
      console.error('Failed to load monitoring data:', err);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 10000);

    // WebSocket for real-time updates
    const ws = createMonitoringSocket((data: any) => {
      setLiveEvents(prev => [data, ...prev].slice(0, 100));
      setWsConnected(true);

      // Auto-refresh stats on workflow events
      if (data.type === 'workflow_complete' || data.type === 'agent_output') {
        fetchData();
      }
    });

    ws.onopen = () => setWsConnected(true);
    ws.onclose = () => setWsConnected(false);

    return () => {
      clearInterval(interval);
      ws.close();
    };
  }, []);

  return (
    <div className="p-8 max-w-6xl mx-auto">
      <div className="flex items-center justify-between mb-10">
        <div>
          <h1 className="text-[32px] font-semibold text-white tracking-tight">Monitoring</h1>
          <p className="text-[15px] text-white/40 mt-1">Real-time agent activity and metrics</p>
        </div>
        <div className="flex items-center gap-3">
          <div className={`flex items-center gap-2 text-[13px] ${wsConnected ? 'text-green-400' : 'text-red-400'}`}>
            {wsConnected ? <Wifi className="w-4 h-4" /> : <WifiOff className="w-4 h-4" />}
            {wsConnected ? 'Connected' : 'Disconnected'}
          </div>
          <button onClick={fetchData}
            className="btn-secondary flex items-center gap-2">
            <RefreshCw className="w-4 h-4" /> Refresh
          </button>
        </div>
      </div>

      {/* Stats */}
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          <div className="glass rounded-2xl p-5">
            <div className="flex items-center gap-2 text-[13px] text-white/40 mb-2">
              <div className="w-7 h-7 rounded-lg bg-green-500/10 flex items-center justify-center">
                <Activity className="w-3.5 h-3.5 text-green-400" strokeWidth={1.8} />
              </div>
              Active Agents
            </div>
            <p className="text-[28px] font-semibold text-white tracking-tight">{stats.active_agents}</p>
          </div>
          <div className="glass rounded-2xl p-5">
            <div className="flex items-center gap-2 text-[13px] text-white/40 mb-2">
              <div className="w-7 h-7 rounded-lg bg-purple-500/10 flex items-center justify-center">
                <MessageSquare className="w-3.5 h-3.5 text-purple-400" strokeWidth={1.8} />
              </div>
              Messages
            </div>
            <p className="text-[28px] font-semibold text-white tracking-tight">{stats.total_messages}</p>
          </div>
          <div className="glass rounded-2xl p-5">
            <div className="flex items-center gap-2 text-[13px] text-white/40 mb-2">
              <div className="w-7 h-7 rounded-lg bg-cyan-500/10 flex items-center justify-center">
                <Zap className="w-3.5 h-3.5 text-cyan-400" strokeWidth={1.8} />
              </div>
              Tokens Used
            </div>
            <p className="text-[28px] font-semibold text-white tracking-tight">{stats.total_tokens.toLocaleString()}</p>
          </div>
          <div className="glass rounded-2xl p-5">
            <div className="flex items-center gap-2 text-[13px] text-white/40 mb-2">
              <div className="w-7 h-7 rounded-lg bg-yellow-500/10 flex items-center justify-center">
                <DollarSign className="w-3.5 h-3.5 text-yellow-400" strokeWidth={1.8} />
              </div>
              Total Cost
            </div>
            <p className="text-[28px] font-semibold text-white tracking-tight">${stats.total_cost.toFixed(4)}</p>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* Live Events */}
        <div className="glass rounded-2xl">
          <div className="px-5 py-4 border-b border-white/[0.06]">
            <h2 className="text-[15px] font-semibold text-white flex items-center gap-2">
              <Activity className="w-4 h-4 text-green-400" strokeWidth={1.8} />
              Live Events
              {liveEvents.length > 0 && (
                <span className="text-[11px] bg-green-500/15 text-green-400 px-2 py-0.5 rounded-full">
                  {liveEvents.length}
                </span>
              )}
            </h2>
          </div>
          <div className="max-h-96 overflow-y-auto p-4 space-y-2">
            {liveEvents.length === 0 ? (
              <p className="text-white/30 text-[13px] text-center py-8">
                Waiting for events... Execute a workflow to see real-time activity.
              </p>
            ) : (
              liveEvents.map((event, i) => (
                <div key={i} className="p-3 bg-white/[0.03] rounded-xl border border-white/[0.04]">
                  <div className="flex items-center gap-2 mb-1">
                    <span className={`text-[11px] px-2 py-0.5 rounded-md font-mono ${
                      event.type === 'agent_output' ? 'bg-blue-500/15 text-blue-400' :
                      event.type === 'workflow_complete' ? 'bg-green-500/15 text-green-400' :
                      event.type === 'workflow_error' ? 'bg-red-500/15 text-red-400' :
                      event.type === 'telegram_message' ? 'bg-purple-500/15 text-purple-400' :
                      'bg-white/[0.06] text-white/40'
                    }`}>
                      {event.type}
                    </span>
                    {event.data?.agent_name && (
                      <span className="text-[11px] text-white/40">{event.data.agent_name}</span>
                    )}
                  </div>
                  {event.data?.content && (
                    <p className="text-[13px] text-white/60 line-clamp-3">{event.data.content}</p>
                  )}
                  {event.data?.tokens_used !== undefined && (
                    <p className="text-[11px] text-white/25 mt-1">
                      {event.data.tokens_used} tokens · ${event.data.cost?.toFixed(4)}
                    </p>
                  )}
                  {event.data?.error && (
                    <p className="text-[13px] text-red-400">{event.data.error}</p>
                  )}
                </div>
              ))
            )}
          </div>
        </div>

        {/* Message History */}
        <div className="glass rounded-2xl">
          <div className="px-5 py-4 border-b border-white/[0.06]">
            <h2 className="text-[15px] font-semibold text-white flex items-center gap-2">
              <MessageSquare className="w-4 h-4 text-purple-400" strokeWidth={1.8} />
              Message History
            </h2>
          </div>
          <div className="max-h-96 overflow-y-auto p-4 space-y-2">
            {messages.length === 0 ? (
              <p className="text-white/30 text-[13px] text-center py-8">
                No messages yet. Run a workflow or send a message via Telegram.
              </p>
            ) : (
              messages.map(msg => (
                <div key={msg.id} className="p-3 bg-white/[0.03] rounded-xl">
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-2">
                      <span className={`text-[11px] px-2 py-0.5 rounded-md ${
                        msg.message_type === 'agent_output' ? 'bg-blue-500/15 text-blue-400' :
                        msg.message_type === 'user_input' ? 'bg-green-500/15 text-green-400' :
                        msg.message_type === 'system' ? 'bg-yellow-500/15 text-yellow-400' :
                        msg.message_type === 'condition' ? 'bg-orange-500/15 text-orange-400' :
                        'bg-white/[0.06] text-white/40'
                      }`}>
                        {msg.message_type}
                      </span>
                      {(msg as any).from_agent_name && (
                        <span className="text-[11px] font-medium text-blue-400">{(msg as any).from_agent_name}</span>
                      )}
                      {msg.channel && msg.channel !== 'internal' && (
                        <span className="text-[11px] text-purple-400">{msg.channel}</span>
                      )}
                    </div>
                    <span className="text-[11px] text-white/20">
                      {new Date(msg.created_at).toLocaleTimeString()}
                    </span>
                  </div>
                  <p className="text-[13px] text-white/60 line-clamp-3">{msg.content}</p>
                  {msg.tokens_used > 0 && (
                    <p className="text-[11px] text-white/25 mt-1">
                      {msg.tokens_used} tokens · ${msg.cost.toFixed(4)}
                    </p>
                  )}
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
