import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom';
import {
  Bot, GitBranch, Activity, LayoutTemplate, Home,
} from 'lucide-react';
import Dashboard from './pages/Dashboard';
import AgentBuilder from './pages/AgentBuilder';
import WorkflowBuilder from './pages/WorkflowBuilder';
import Monitoring from './pages/Monitoring';
import Templates from './pages/Templates';

const navItems = [
  { to: '/', icon: Home, label: 'Dashboard' },
  { to: '/agents', icon: Bot, label: 'Agents' },
  { to: '/workflows', icon: GitBranch, label: 'Workflows' },
  { to: '/monitoring', icon: Activity, label: 'Monitoring' },
  { to: '/templates', icon: LayoutTemplate, label: 'Templates' },
];

export default function App() {
  return (
    <BrowserRouter>
      <div className="flex h-screen overflow-hidden bg-mesh">
        {/* Sidebar */}
        <nav className="w-[220px] flex flex-col border-r border-white/[0.06] bg-black/40 backdrop-blur-xl">
          <div className="px-5 pt-6 pb-5">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                <Bot className="w-4.5 h-4.5 text-white" />
              </div>
              <div>
                <h1 className="text-[15px] font-semibold text-white tracking-tight">Cortex</h1>
              </div>
            </div>
          </div>
          <div className="flex-1 px-3 space-y-0.5">
            {navItems.map(({ to, icon: Icon, label }) => (
              <NavLink
                key={to}
                to={to}
                end={to === '/'}
                className={({ isActive }) =>
                  `flex items-center gap-2.5 px-3 py-2 rounded-lg text-[13px] font-medium tracking-[-0.01em] transition-all duration-200 ${
                    isActive
                      ? 'bg-white/[0.1] text-white'
                      : 'text-white/50 hover:text-white/80 hover:bg-white/[0.04]'
                  }`
                }
              >
                <Icon className="w-[18px] h-[18px]" strokeWidth={1.8} />
                {label}
              </NavLink>
            ))}
          </div>
          <div className="px-5 py-4">
            <p className="text-[11px] text-white/20 tracking-wide">Cortex v1.0</p>
          </div>
        </nav>

        {/* Main content */}
        <main className="flex-1 overflow-y-auto">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/agents" element={<AgentBuilder />} />
            <Route path="/workflows" element={<WorkflowBuilder />} />
            <Route path="/monitoring" element={<Monitoring />} />
            <Route path="/templates" element={<Templates />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
