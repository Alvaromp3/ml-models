import { ChevronRight, AlertTriangle, TrendingUp, Users } from 'lucide-react';
import { Link } from 'react-router-dom';
import type { Player } from '../../types';

interface PlayerListProps {
  title: string;
  players: Player[];
  type: 'risk' | 'top';
  viewAllLink?: string;
}

export default function PlayerList({ title, players, type, viewAllLink }: PlayerListProps) {
  const config = {
    risk: {
      gradient: 'from-orange-500 to-red-600',
      shadow: 'shadow-red-500/20',
      icon: AlertTriangle,
    },
    top: {
      gradient: 'from-cyan-500 to-blue-600',
      shadow: 'shadow-cyan-500/20',
      icon: TrendingUp,
    },
  };

  const { gradient, shadow, icon: Icon } = config[type];

  return (
    <div 
      className="card p-6 opacity-0 animate-slide-in-up"
      style={{ animationDelay: type === 'risk' ? '250ms' : '300ms', animationFillMode: 'forwards' }}
    >
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-3">
          <div className={`p-2.5 rounded-xl bg-gradient-to-br ${gradient} ${shadow} shadow-lg`}>
            <Icon className="w-5 h-5 text-white" />
          </div>
          <h3 className="text-base font-semibold text-white">{title}</h3>
        </div>
        {viewAllLink && (
          <Link 
            to={viewAllLink}
            className="flex items-center gap-1 text-xs font-medium text-cyan-400 hover:text-cyan-300 transition-colors group"
          >
            View all
            <ChevronRight className="w-4 h-4 group-hover:translate-x-0.5 transition-transform" />
          </Link>
        )}
      </div>

      <div className="space-y-2">
        {players.length === 0 ? (
          <div className="text-center py-8">
            <div className="w-12 h-12 mx-auto mb-3 rounded-xl bg-slate-800/50 flex items-center justify-center">
              <Users className="w-6 h-6 text-slate-600" />
            </div>
            <p className="text-slate-500 text-sm">No players to display</p>
          </div>
        ) : (
          players.slice(0, 5).map((player, index) => (
            <div 
              key={player.id}
              className="flex items-center gap-3 p-3 bg-slate-800/30 hover:bg-slate-800/60 rounded-xl transition-all duration-200 cursor-pointer group"
            >
              {/* Player number */}
              <div className={`
                w-10 h-10 rounded-lg flex items-center justify-center font-bold text-sm
                ${type === 'risk' 
                  ? 'bg-red-500/15 text-red-400 border border-red-500/30' 
                  : 'bg-cyan-500/15 text-cyan-400 border border-cyan-500/30'
                }
              `}>
                {player.number}
              </div>

              {/* Player info */}
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-white truncate group-hover:text-cyan-400 transition-colors">
                  {player.name}
                </p>
                <p className="text-xs text-slate-500">#{player.number}</p>
              </div>

              {/* Risk badge or stats */}
              {type === 'risk' ? (
                <div className={`
                  px-2 py-1 rounded-md text-[10px] font-semibold uppercase
                  ${player.riskLevel === 'high' 
                    ? 'bg-red-500/15 text-red-400' 
                    : player.riskLevel === 'medium' 
                    ? 'bg-yellow-500/15 text-yellow-400' 
                    : 'bg-emerald-500/15 text-emerald-400'
                  }
                `}>
                  {player.riskLevel}
                </div>
              ) : (
                <div className="text-right">
                  <p className="text-sm font-semibold text-white">{player.avgLoad}</p>
                  <p className="text-[10px] text-slate-500">load</p>
                </div>
              )}

              <ChevronRight className="w-4 h-4 text-slate-600 group-hover:text-cyan-400 group-hover:translate-x-0.5 transition-all" />
            </div>
          ))
        )}
      </div>
    </div>
  );
}
