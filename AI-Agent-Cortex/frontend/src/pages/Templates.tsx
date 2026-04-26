import { useEffect, useState } from 'react';
import { LayoutTemplate, ArrowRight, CheckCircle, Plus, FileText, Users } from 'lucide-react';
import { templatesApi } from '../services/api';
import { Template } from '../types';
import { useNavigate } from 'react-router-dom';

export default function Templates() {
  const [templates, setTemplates] = useState<Template[]>([]);
  const [loading, setLoading] = useState(true);
  const [instantiating, setInstantiating] = useState<string | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    templatesApi.list()
      .then(setTemplates)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  const handleInstantiate = async (templateId: string) => {
    setInstantiating(templateId);
    try {
      const result = await templatesApi.instantiate(templateId);
      alert(`${result.message}\n\nCreated ${result.agents_created.length} agents.`);
      navigate('/workflows');
    } catch (err) {
      alert(`Error: ${err}`);
    } finally {
      setInstantiating(null);
    }
  };

  return (
    <div className="p-8 max-w-6xl mx-auto">
      <div className="flex items-center justify-between mb-10">
        <div>
          <h1 className="text-[32px] font-semibold text-white tracking-tight">Templates</h1>
          <p className="text-[15px] text-white/40 mt-1">Pre-built workflows to get started quickly</p>
        </div>
      </div>

      {loading ? (
        <div className="flex items-center justify-center h-64">
          <div className="w-6 h-6 border-2 border-white/10 border-t-white/60 rounded-full animate-spin" />
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          {templates.map(template => (
            <div
              key={template.id}
              className="glass glass-hover rounded-2xl p-6 transition-all duration-300"
            >
              <div className="flex items-start gap-4 mb-4">
                <div className="w-10 h-10 rounded-xl bg-purple-500/10 flex items-center justify-center flex-shrink-0">
                  <LayoutTemplate className="w-5 h-5 text-purple-400" strokeWidth={1.8} />
                </div>
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <h3 className="text-[15px] font-semibold text-white">{template.name}</h3>
                    {template.is_builtin && (
                      <span className="text-[11px] px-2 py-0.5 bg-blue-500/15 text-blue-400 rounded-full border border-blue-500/20">Built-in</span>
                    )}
                  </div>
                  <p className="text-[13px] text-white/40 mt-1">{template.description}</p>
                </div>
              </div>

              {/* Template agents preview */}
              <div className="mb-4">
                <p className="text-[11px] font-medium text-white/30 uppercase tracking-wider mb-2">Agents</p>
                <div className="space-y-1.5">
                  {template.agents.map((agent: any, i: number) => (
                    <div key={i} className="flex items-center gap-2 p-2 bg-white/[0.03] rounded-lg border border-white/[0.04]">
                      <Users className="w-3.5 h-3.5 text-blue-400" strokeWidth={1.8} />
                      <span className="text-[13px] text-white/70">{agent.name}</span>
                      <span className="text-[11px] text-white/30">({agent.role})</span>
                      {agent.tools?.map((t: string) => (
                        <span key={t} className="text-[10px] px-1.5 py-0.5 bg-white/[0.04] text-white/30 rounded-md border border-white/[0.06]">
                          {t}
                        </span>
                      ))}
                    </div>
                  ))}
                </div>
              </div>

              {/* Graph preview */}
              <div className="mb-5">
                <p className="text-[11px] font-medium text-white/30 uppercase tracking-wider mb-2">Flow</p>
                <div className="flex flex-wrap items-center gap-1">
                  {template.graph.nodes
                    .sort((a, b) => a.position.y - b.position.y)
                    .map((node, i) => (
                      <div key={node.id} className="flex items-center gap-1">
                        <span className={`text-[11px] px-2 py-1 rounded-lg ${
                          node.type === 'start' ? 'bg-green-500/10 text-green-400 border border-green-500/20' :
                          node.type === 'end' ? 'bg-red-500/10 text-red-400 border border-red-500/20' :
                          node.type === 'condition' ? 'bg-yellow-500/10 text-yellow-400 border border-yellow-500/20' :
                          'bg-blue-500/10 text-blue-400 border border-blue-500/20'
                        }`}>
                          {node.label}
                        </span>
                        {i < template.graph.nodes.length - 1 && (
                          <ArrowRight className="w-3 h-3 text-white/15" />
                        )}
                      </div>
                    ))}
                </div>
              </div>

              <button
                onClick={() => handleInstantiate(template.id)}
                disabled={instantiating === template.id}
                className="btn-primary w-full flex items-center justify-center gap-2 disabled:opacity-40 disabled:cursor-not-allowed"
              >
                {instantiating === template.id ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white/20 border-t-white rounded-full animate-spin" />
                    Creating...
                  </>
                ) : (
                  <>
                    <Plus className="w-4 h-4" />
                    Use Template
                  </>
                )}
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
