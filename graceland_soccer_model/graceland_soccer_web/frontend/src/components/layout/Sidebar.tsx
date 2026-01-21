import { NavLink } from 'react-router-dom';
import { 
  LayoutDashboard, 
  Users, 
  BarChart3, 
  Dumbbell, 
  Settings,
  X,
  ChevronRight,
  Trophy,
  FileSearch
} from 'lucide-react';

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
}

const navItems = [
  { path: '/', icon: LayoutDashboard, label: 'Dashboard', description: 'Overview & KPIs' },
  { path: '/players', icon: Users, label: 'Players', description: 'Team roster' },
  { path: '/analysis', icon: BarChart3, label: 'Analysis', description: 'Risk predictions' },
  { path: '/lineup', icon: Trophy, label: 'Best Lineup', description: 'Optimal XI' },
  { path: '/training', icon: Dumbbell, label: 'Training', description: 'ML models' },
  { path: '/data-audit', icon: FileSearch, label: 'Data Audit', description: 'Clean outliers' },
  { path: '/settings', icon: Settings, label: 'Settings', description: 'Preferences' },
];

export default function Sidebar({ isOpen, onClose }: SidebarProps) {
  return (
    <>
      {/* Mobile overlay */}
      {isOpen && (
        <div 
          className="fixed inset-0 bg-black/70 backdrop-blur-sm z-40 lg:hidden"
          onClick={onClose}
        />
      )}
      
      {/* Sidebar */}
      <aside className={`
        fixed top-0 left-0 z-50 h-full w-72 
        bg-[#0c1222] border-r border-slate-800/80
        transform transition-transform duration-300 ease-in-out
        lg:translate-x-0 lg:static lg:z-auto
        flex flex-col
        ${isOpen ? 'translate-x-0' : '-translate-x-full'}
      `}>
        {/* Logo Section */}
        <div className="p-6 border-b border-slate-800/80">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <img 
                src="/graceland-logo.png" 
                alt="Graceland" 
                className="w-12 h-12 object-contain"
              />
              <div>
                <h1 className="font-bold text-lg text-white">Graceland Soccer</h1>
                <p className="text-[11px] text-slate-500 font-medium tracking-wide">UNIVERSITY</p>
              </div>
            </div>
            <button 
              onClick={onClose}
              className="lg:hidden p-2 hover:bg-slate-800 rounded-lg transition-colors"
            >
              <X className="w-5 h-5 text-slate-400" />
            </button>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-4 overflow-y-auto">
          <p className="text-[10px] font-bold text-slate-600 uppercase tracking-[0.2em] mb-4 px-3">
            Main Menu
          </p>
          <div className="space-y-1.5">
            {navItems.map((item) => (
              <NavLink
                key={item.path}
                to={item.path}
                onClick={onClose}
                className={({ isActive }) => `
                  flex items-center gap-3 px-3 py-3 rounded-xl transition-all duration-200 group
                  ${isActive 
                    ? 'bg-slate-800/60 text-white border border-slate-700/50' 
                    : 'text-slate-400 hover:bg-slate-800/50 hover:text-white'
                  }
                `}
              >
                {({ isActive }) => (
                  <>
                    <div className={`
                      p-2 rounded-lg transition-all
                      ${isActive 
                        ? 'bg-slate-700/80' 
                        : 'bg-slate-800/80 group-hover:bg-slate-700'
                      }
                    `}>
                      <item.icon className={`w-4 h-4 ${isActive ? 'text-white' : 'text-slate-400 group-hover:text-white'}`} />
                    </div>
                    <div className="flex-1">
                      <p className={`text-sm font-medium ${isActive ? 'text-white' : ''}`}>{item.label}</p>
                      <p className="text-[10px] text-slate-500">{item.description}</p>
                    </div>
                    {isActive && (
                      <ChevronRight className="w-4 h-4 text-slate-400" />
                    )}
                  </>
                )}
              </NavLink>
            ))}
          </div>
        </nav>

      </aside>
    </>
  );
}
