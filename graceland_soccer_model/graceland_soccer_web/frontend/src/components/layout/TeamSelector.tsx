import { Users, UserCheck } from 'lucide-react';
import { useTeam } from '../../contexts/TeamContext';

export default function TeamSelector() {
  const { currentTeam, switchTeam, teamStatus } = useTeam();

  return (
    <div className="flex items-center gap-2 bg-slate-800/50 border border-slate-700/50 rounded-xl p-1">
      <button
        onClick={() => switchTeam('mens')}
        className={`
          flex items-center gap-2 px-4 py-2 rounded-lg transition-all duration-200
          ${currentTeam === 'mens'
            ? 'bg-blue-600 text-white shadow-lg shadow-blue-600/20'
            : 'text-slate-400 hover:text-white hover:bg-slate-700/50'
          }
        `}
      >
        <Users className={`w-4 h-4 ${currentTeam === 'mens' ? 'text-white' : 'text-slate-500'}`} />
        <span className="text-sm font-medium">Men's</span>
        {teamStatus?.mens?.loaded && (
          <span className={`text-xs px-1.5 py-0.5 rounded ${currentTeam === 'mens' ? 'bg-blue-700' : 'bg-slate-700'}`}>
            {teamStatus.mens.rowCount}
          </span>
        )}
      </button>
      
      <button
        onClick={() => switchTeam('womens')}
        className={`
          flex items-center gap-2 px-4 py-2 rounded-lg transition-all duration-200
          ${currentTeam === 'womens'
            ? 'bg-pink-600 text-white shadow-lg shadow-pink-600/20'
            : 'text-slate-400 hover:text-white hover:bg-slate-700/50'
          }
        `}
      >
        <UserCheck className={`w-4 h-4 ${currentTeam === 'womens' ? 'text-white' : 'text-slate-500'}`} />
        <span className="text-sm font-medium">Women's</span>
        {teamStatus?.womens?.loaded && (
          <span className={`text-xs px-1.5 py-0.5 rounded ${currentTeam === 'womens' ? 'bg-pink-700' : 'bg-slate-700'}`}>
            {teamStatus.womens.rowCount}
          </span>
        )}
      </button>
    </div>
  );
}
