import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { 
  Trophy, 
  Zap, 
  Gauge, 
  Activity, 
  TrendingUp, 
  Target,
  Award,
  Medal,
  Flame,
  Bolt,
  Wind,
  BarChart3
} from 'lucide-react';
import { playersApi } from '../services/api';

interface RankingMetric {
  id: string;
  name: string;
  description: string;
  icon: any;
  color: string;
  unit: string;
  key: string;
}

const metrics: RankingMetric[] = [
  {
    id: 'player_load',
    name: 'Player Load',
    description: 'Average player load',
    icon: Zap,
    color: 'text-blue-400',
    unit: 'units',
    key: 'player_load'
  },
  {
    id: 'total_distance',
    name: 'Total Distance',
    description: 'Who runs the most (total)',
    icon: Activity,
    color: 'text-green-400',
    unit: 'miles',
    key: 'total_distance'
  },
  {
    id: 'distance',
    name: 'Average Distance',
    description: 'Average distance per session',
    icon: TrendingUp,
    color: 'text-emerald-400',
    unit: 'miles',
    key: 'distance'
  },
  {
    id: 'total_sprints',
    name: 'Total Sprints',
    description: 'Total sprints performed',
    icon: Bolt,
    color: 'text-yellow-400',
    unit: 'yards',
    key: 'total_sprints'
  },
  {
    id: 'sprint_distance',
    name: 'Sprint Distance',
    description: 'Average distance in sprints',
    icon: Wind,
    color: 'text-orange-400',
    unit: 'yards',
    key: 'sprint_distance'
  },
  {
    id: 'max_speed',
    name: 'Maximum Speed',
    description: 'Maximum speed reached',
    icon: Gauge,
    color: 'text-red-400',
    unit: 'mph',
    key: 'max_speed'
  },
  {
    id: 'top_speed',
    name: 'Average Speed',
    description: 'Average maximum speed',
    icon: BarChart3,
    color: 'text-purple-400',
    unit: 'mph',
    key: 'top_speed'
  },
  {
    id: 'work_ratio',
    name: 'Intensity (Work Ratio)',
    description: 'Average work ratio',
    icon: Flame,
    color: 'text-pink-400',
    unit: '',
    key: 'work_ratio'
  },
  {
    id: 'max_intensity',
    name: 'Maximum Intensity',
    description: 'Maximum intensity reached',
    icon: Target,
    color: 'text-rose-400',
    unit: '',
    key: 'max_intensity'
  },
  {
    id: 'total_energy',
    name: 'Total Energy',
    description: 'Total energy consumed',
    icon: Zap,
    color: 'text-cyan-400',
    unit: 'kcal',
    key: 'total_energy'
  },
  {
    id: 'energy',
    name: 'Average Energy',
    description: 'Average energy per session',
    icon: Activity,
    color: 'text-teal-400',
    unit: 'kcal',
    key: 'energy'
  },
  {
    id: 'max_power',
    name: 'Maximum Power',
    description: 'Maximum power generated',
    icon: Bolt,
    color: 'text-amber-400',
    unit: 'w/kg',
    key: 'max_power'
  },
  {
    id: 'power_score',
    name: 'Average Power',
    description: 'Average relative power',
    icon: Trophy,
    color: 'text-indigo-400',
    unit: 'w/kg',
    key: 'power_score'
  },
  {
    id: 'max_acceleration',
    name: 'Maximum Acceleration',
    description: 'Maximum acceleration reached',
    icon: TrendingUp,
    color: 'text-lime-400',
    unit: 'yd/s²',
    key: 'max_acceleration'
  },
  {
    id: 'distance_per_min',
    name: 'Distance per Minute',
    description: 'Average work pace',
    icon: BarChart3,
    color: 'text-sky-400',
    unit: 'yd/min',
    key: 'distance_per_min'
  },
  {
    id: 'total_impacts',
    name: 'Total Impacts',
    description: 'Total impacts received',
    icon: Target,
    color: 'text-violet-400',
    unit: '',
    key: 'total_impacts'
  }
];

function getRankIcon(rank: number) {
  if (rank === 1) return <Trophy className="w-5 h-5 text-yellow-400" />;
  if (rank === 2) return <Medal className="w-5 h-5 text-gray-300" />;
  if (rank === 3) return <Medal className="w-5 h-5 text-amber-600" />;
  return <Award className="w-5 h-5 text-slate-500" />;
}

function formatValue(value: number, unit: string): string {
  if (value === 0) return '0';
  if (value < 1) return value.toFixed(2);
  if (value < 10) return value.toFixed(1);
  return Math.round(value).toLocaleString();
}

export default function Rankings() {
  const [selectedMetric, setSelectedMetric] = useState<string>('player_load');
  
  const currentMetric = metrics.find(m => m.id === selectedMetric) || metrics[0];
  
  const { data: rankings, isLoading } = useQuery({
    queryKey: ['rankings', selectedMetric],
    queryFn: () => playersApi.getRankings(selectedMetric),
  });

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-slate-900 to-slate-800 rounded-xl p-6 border border-slate-700">
        <div className="flex items-center gap-3 mb-2">
          <div className="p-3 bg-yellow-600/20 rounded-lg">
            <Trophy className="w-6 h-6 text-yellow-400" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-white">Player Rankings</h1>
            <p className="text-slate-400">Statistics and rankings by metrics</p>
          </div>
        </div>
      </div>

      {/* Metric Selector */}
      <div className="bg-slate-900/50 rounded-xl p-6 border border-slate-700">
        <h2 className="text-lg font-semibold text-white mb-4">Select Metric</h2>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
          {metrics.map((metric) => {
            const Icon = metric.icon;
            const isSelected = selectedMetric === metric.id;
            return (
              <button
                key={metric.id}
                onClick={() => setSelectedMetric(metric.id)}
                className={`
                  p-4 rounded-lg border transition-all duration-200 text-left
                  ${isSelected 
                    ? 'bg-slate-800 border-cyan-500/50 shadow-lg shadow-cyan-500/10' 
                    : 'bg-slate-800/50 border-slate-700 hover:border-slate-600 hover:bg-slate-800'
                  }
                `}
              >
                <div className="flex items-center gap-3 mb-2">
                  <Icon className={`w-5 h-5 ${isSelected ? metric.color : 'text-slate-500'}`} />
                  <span className={`text-sm font-semibold ${isSelected ? 'text-white' : 'text-slate-400'}`}>
                    {metric.name}
                  </span>
                </div>
                <p className="text-xs text-slate-500">{metric.description}</p>
              </button>
            );
          })}
        </div>
      </div>

      {/* Rankings Table */}
      <div className="bg-slate-900/50 rounded-xl border border-slate-700 overflow-hidden">
        <div className="p-6 border-b border-slate-700">
          <div className="flex items-center gap-3">
            {(() => {
              const Icon = currentMetric.icon;
              return <Icon className={`w-6 h-6 ${currentMetric.color}`} />;
            })()}
            <div>
              <h2 className="text-xl font-bold text-white">{currentMetric.name}</h2>
              <p className="text-sm text-slate-400">{currentMetric.description}</p>
            </div>
          </div>
        </div>

        {isLoading ? (
          <div className="p-12 text-center">
            <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-cyan-400"></div>
            <p className="text-slate-400 mt-4">Loading rankings...</p>
          </div>
        ) : !rankings || rankings.length === 0 ? (
          <div className="p-12 text-center">
            <Trophy className="w-12 h-12 text-slate-600 mx-auto mb-4" />
            <p className="text-slate-400">No data available</p>
            <p className="text-slate-500 text-sm mt-2">Load data to see rankings</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-slate-700">
                  <th className="text-left p-4 text-xs font-semibold text-slate-400 uppercase tracking-wider">Rank</th>
                  <th className="text-left p-4 text-xs font-semibold text-slate-400 uppercase tracking-wider">Player</th>
                  <th className="text-right p-4 text-xs font-semibold text-slate-400 uppercase tracking-wider">
                    {currentMetric.name} ({currentMetric.unit})
                  </th>
                  <th className="text-right p-4 text-xs font-semibold text-slate-400 uppercase tracking-wider">Sessions</th>
                </tr>
              </thead>
              <tbody>
                {rankings.map((player: any, index: number) => {
                  const value = player.metrics[currentMetric.key] || 0;
                  const isTopThree = player.rank <= 3;
                  
                  return (
                    <tr 
                      key={player.name}
                      className={`
                        border-b border-slate-800/50 transition-colors
                        ${isTopThree ? 'bg-slate-800/30' : 'hover:bg-slate-800/20'}
                      `}
                    >
                      <td className="p-4">
                        <div className="flex items-center gap-3">
                          {getRankIcon(player.rank)}
                          <span className={`font-bold ${isTopThree ? 'text-white' : 'text-slate-400'}`}>
                            #{player.rank}
                          </span>
                        </div>
                      </td>
                      <td className="p-4">
                        <div className="flex items-center gap-3">
                          <div className={`
                            w-10 h-10 rounded-lg flex items-center justify-center font-bold text-sm
                            ${isTopThree 
                              ? 'bg-gradient-to-br from-yellow-500/20 to-amber-500/20 text-yellow-400 border border-yellow-500/30' 
                              : 'bg-slate-800 text-slate-400 border border-slate-700'
                            }
                          `}>
                            {player.name.charAt(0).toUpperCase()}
                          </div>
                          <span className={`font-semibold ${isTopThree ? 'text-white' : 'text-slate-300'}`}>
                            {player.name}
                          </span>
                        </div>
                      </td>
                      <td className="p-4 text-right">
                        <span className={`text-lg font-bold ${isTopThree ? currentMetric.color : 'text-white'}`}>
                          {formatValue(value, currentMetric.unit)}
                        </span>
                        {currentMetric.unit && (
                          <span className="text-xs text-slate-500 ml-1">{currentMetric.unit}</span>
                        )}
                      </td>
                      <td className="p-4 text-right">
                        <span className="text-slate-400">{player.metrics.sessions || 0}</span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Additional Stats */}
      {rankings && rankings.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-slate-900/50 p-4 rounded-lg border border-slate-700">
            <p className="text-xs text-slate-500 mb-1">Best</p>
            <p className="text-lg font-bold text-white">
              {rankings[0]?.name}
            </p>
            <p className="text-sm text-cyan-400">
              {formatValue(rankings[0]?.metrics[currentMetric.key] || 0, currentMetric.unit)} {currentMetric.unit}
            </p>
          </div>
          <div className="bg-slate-900/50 p-4 rounded-lg border border-slate-700">
            <p className="text-xs text-slate-500 mb-1">Average</p>
            <p className="text-lg font-bold text-white">
              {rankings.length > 0 
                ? formatValue(
                    rankings.reduce((sum: number, p: any) => sum + (p.metrics[currentMetric.key] || 0), 0) / rankings.length,
                    currentMetric.unit
                  )
                : '0'
              }
            </p>
            <p className="text-sm text-slate-400">{currentMetric.unit}</p>
          </div>
          <div className="bg-slate-900/50 p-4 rounded-lg border border-slate-700">
            <p className="text-xs text-slate-500 mb-1">Total Players</p>
            <p className="text-lg font-bold text-white">{rankings.length}</p>
            <p className="text-sm text-slate-400">in ranking</p>
          </div>
        </div>
      )}
    </div>
  );
}
