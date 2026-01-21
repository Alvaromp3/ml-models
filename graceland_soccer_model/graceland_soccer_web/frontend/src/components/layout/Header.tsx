import { Menu, Bell, Search, Calendar } from 'lucide-react';

interface HeaderProps {
  onMenuClick: () => void;
}

export default function Header({ onMenuClick }: HeaderProps) {
  const today = new Date().toLocaleDateString('en-US', {
    weekday: 'long',
    year: 'numeric',
    month: 'long',
    day: 'numeric'
  });

  return (
    <header className="sticky top-0 z-30 bg-[#080d19]/80 backdrop-blur-xl border-b border-slate-800/50">
      <div className="flex items-center justify-between px-6 py-4">
        {/* Left side */}
        <div className="flex items-center gap-4">
          <button
            onClick={onMenuClick}
            className="lg:hidden p-2.5 bg-slate-800/50 hover:bg-slate-700 rounded-xl transition-colors"
          >
            <Menu className="w-5 h-5 text-slate-300" />
          </button>
          
          {/* Search */}
          <div className="hidden md:flex items-center gap-3 bg-slate-800/50 border border-slate-700/50 rounded-xl px-4 py-2.5 w-80 focus-within:border-slate-600 focus-within:bg-slate-800 transition-all">
            <Search className="w-4 h-4 text-slate-500" />
            <input
              type="text"
              placeholder="Search players, sessions..."
              className="bg-transparent border-none outline-none text-sm text-white placeholder-slate-500 w-full"
            />
            <div className="flex items-center gap-1 px-2 py-1 bg-slate-700/50 rounded-md">
              <span className="text-[10px] text-slate-400 font-medium">⌘K</span>
            </div>
          </div>
        </div>

        {/* Right side */}
        <div className="flex items-center gap-3">
          {/* Date */}
          <div className="hidden lg:flex items-center gap-2.5 px-4 py-2.5 bg-slate-800/50 border border-slate-700/50 rounded-xl">
            <Calendar className="w-4 h-4 text-slate-400" />
            <span className="text-sm text-slate-300 font-medium">{today}</span>
          </div>

          {/* Notifications */}
          <button className="relative p-2.5 bg-slate-800/50 border border-slate-700/50 rounded-xl hover:bg-slate-700 hover:border-slate-600 transition-all group">
            <Bell className="w-5 h-5 text-slate-400 group-hover:text-white transition-colors" />
            <span className="absolute top-2 right-2 w-2 h-2 bg-red-500 rounded-full animate-pulse" />
          </button>
        </div>
      </div>
    </header>
  );
}
