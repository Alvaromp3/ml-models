import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { 
  Search, 
  AlertTriangle, 
  Activity, 
  ChevronRight, 
  UserMinus, 
  UserPlus,
  X, 
  Users,
  TrendingUp,
  Zap,
  BarChart3,
  EyeOff,
  Eye
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { playersApi, useDataStatus } from '../services/api';
import type { Player } from '../types';

const riskColors = {
  low: { bg: 'bg-emerald-500/15', text: 'text-emerald-400', border: 'border-emerald-500/30', color: '#22c55e' },
  medium: { bg: 'bg-yellow-500/15', text: 'text-yellow-400', border: 'border-yellow-500/30', color: '#eab308' },
  high: { bg: 'bg-red-500/15', text: 'text-red-400', border: 'border-red-500/30', color: '#ef4444' },
};

function PlayerCard({ player, onExclude }: { player: Player; onExclude: (name: string) => void }) {
  const colors = riskColors[player.riskLevel];
  
  return (
    <div className="card card-hover p-5 group relative">
      {/* Exclude button */}
      <button
        onClick={(e) => {
          e.stopPropagation();
          onExclude(player.name);
        }}
        className="absolute top-3 right-3 p-2 bg-orange-500/10 hover:bg-orange-500/20 border border-orange-500/20 rounded-lg opacity-0 group-hover:opacity-100 transition-all z-10"
        title="Exclude from analysis"
      >
        <UserMinus className="w-4 h-4 text-orange-400" />
      </button>

      <div className="flex items-start justify-between pr-10">
        <div className="flex items-center gap-4">
          <div className={`w-14 h-14 rounded-xl flex items-center justify-center font-bold text-2xl ${colors.bg} ${colors.text} border ${colors.border}`}>
            {player.number}
          </div>
          <div>
            <h3 className="font-semibold text-white text-lg group-hover:text-slate-300 transition-colors">
              {player.name}
            </h3>
            <p className="text-slate-500 text-sm">#{player.number}</p>
          </div>
        </div>
        <div className={`px-2.5 py-1 rounded-lg text-xs font-semibold uppercase flex items-center gap-1.5 ${colors.bg} ${colors.text} border ${colors.border}`}>
          <span className={`w-1.5 h-1.5 rounded-full ${player.riskLevel === 'high' ? 'bg-red-400' : player.riskLevel === 'medium' ? 'bg-yellow-400' : 'bg-emerald-400'}`} />
          {player.riskLevel}
        </div>
      </div>

      <div className="mt-5 grid grid-cols-3 gap-3">
        <div className="text-center p-3 bg-slate-800/30 rounded-lg">
          <Zap className="w-4 h-4 text-slate-400 mx-auto mb-1" />
          <p className="text-xl font-bold text-white">{player.avgLoad}</p>
          <p className="text-[10px] text-slate-500">Avg Load</p>
        </div>
        <div className="text-center p-3 bg-slate-800/30 rounded-lg">
          <TrendingUp className="w-4 h-4 text-emerald-400 mx-auto mb-1" />
          <p className="text-xl font-bold text-white">{player.avgSpeed}</p>
          <p className="text-[10px] text-slate-500">Avg Speed</p>
        </div>
        <div className="text-center p-3 bg-slate-800/30 rounded-lg">
          <Activity className="w-4 h-4 text-purple-400 mx-auto mb-1" />
          <p className="text-xl font-bold text-white">{player.sessions}</p>
          <p className="text-[10px] text-slate-500">Sessions</p>
        </div>
      </div>

      <div className="mt-4 flex items-center justify-between text-sm pt-3 border-t border-slate-800/50">
        <div className="flex items-center gap-2 text-slate-500">
          <Activity className="w-4 h-4" />
          <span>Last: {player.lastSession || 'N/A'}</span>
        </div>
        <a 
          href={`/analysis?player=${player.id}`}
          className="flex items-center gap-1 text-slate-400 hover:text-slate-300 transition-colors"
        >
          Analyze
          <ChevronRight className="w-4 h-4" />
        </a>
      </div>
    </div>
  );
}

export default function Players() {
  const queryClient = useQueryClient();
  const [search, setSearch] = useState('');
  const [riskFilter, setRiskFilter] = useState<string>('all');
  const [showExcluded, setShowExcluded] = useState(false);
  const [confirmExclude, setConfirmExclude] = useState<{ open: boolean; playerName: string }>({
    open: false,
    playerName: '',
  });

  const { data: dataStatus } = useDataStatus();

  const { data: players, isLoading } = useQuery({
    queryKey: ['players'],
    queryFn: playersApi.getAll,
    enabled: !!dataStatus?.loaded,
  });

  const { data: excludedPlayers } = useQuery({
    queryKey: ['players', 'excluded'],
    queryFn: playersApi.getExcluded,
    enabled: !!dataStatus?.loaded,
  });

  const excludeMutation = useMutation({
    mutationFn: playersApi.excludePlayer,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['players'] });
      queryClient.invalidateQueries({ queryKey: ['dashboard'] });
      setConfirmExclude({ open: false, playerName: '' });
    },
  });

  const restoreMutation = useMutation({
    mutationFn: playersApi.restorePlayer,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['players'] });
      queryClient.invalidateQueries({ queryKey: ['dashboard'] });
    },
  });

  const filteredPlayers = players?.filter((player) => {
    const matchesSearch = player.name.toLowerCase().includes(search.toLowerCase());
    const matchesRisk = riskFilter === 'all' || player.riskLevel === riskFilter;
    return matchesSearch && matchesRisk;
  }) ?? [];

  const riskCounts = {
    all: players?.length ?? 0,
    low: players?.filter(p => p.riskLevel === 'low').length ?? 0,
    medium: players?.filter(p => p.riskLevel === 'medium').length ?? 0,
    high: players?.filter(p => p.riskLevel === 'high').length ?? 0,
  };

  const handleExcludeClick = (playerName: string) => {
    setConfirmExclude({ open: true, playerName });
  };

  const confirmExcludePlayer = () => {
    if (confirmExclude.playerName) {
      excludeMutation.mutate(confirmExclude.playerName);
    }
  };

  // Chart data for load distribution
  const loadChartData = filteredPlayers
    .sort((a, b) => b.avgLoad - a.avgLoad)
    .slice(0, 10)
    .map(p => ({
      name: p.name.split(' ')[0],
      load: p.avgLoad,
      risk: p.riskLevel,
    }));

  // No data state
  if (!dataStatus?.loaded) {
    return (
      <div className="min-h-[70vh] flex items-center justify-center animate-fade-in">
        <div className="card p-8 max-w-md text-center">
          <div className="w-16 h-16 mx-auto mb-6 bg-slate-800/60 border border-slate-700/50 rounded-2xl flex items-center justify-center">
            <Users className="w-8 h-8 text-slate-300" />
          </div>
          <h2 className="text-xl font-bold text-white mb-2">No Data Loaded</h2>
          <p className="text-slate-400 text-sm mb-6">
            Load CSV data first to view players.
          </p>
          <a 
            href="/"
            className="inline-flex items-center gap-2 px-5 py-2.5 btn-primary rounded-xl font-medium text-white text-sm"
          >
            Go to Dashboard
            <ChevronRight className="w-4 h-4" />
          </a>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Exclude Confirmation Modal */}
      {confirmExclude.open && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="card p-6 max-w-md w-full animate-slide-in-up">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white">Exclude Player</h3>
              <button
                onClick={() => setConfirmExclude({ open: false, playerName: '' })}
                className="p-2 hover:bg-slate-800 rounded-lg transition-colors"
              >
                <X className="w-5 h-5 text-slate-400" />
              </button>
            </div>
            <div className="mb-6">
              <div className="w-12 h-12 mx-auto mb-4 bg-orange-500/10 rounded-xl flex items-center justify-center">
                <UserMinus className="w-6 h-6 text-orange-400" />
              </div>
              <p className="text-center text-slate-300">
                Exclude <span className="font-semibold text-white">{confirmExclude.playerName}</span> from analysis?
              </p>
              <p className="text-center text-sm text-slate-500 mt-2">
                This player will be hidden from all comparisons and lineups. You can restore them anytime.
              </p>
            </div>
            <div className="flex gap-3">
              <button
                onClick={() => setConfirmExclude({ open: false, playerName: '' })}
                className="flex-1 py-2.5 bg-slate-800 hover:bg-slate-700 rounded-xl font-medium text-white transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={confirmExcludePlayer}
                disabled={excludeMutation.isPending}
                className="flex-1 py-2.5 bg-orange-500 hover:bg-orange-600 rounded-xl font-medium text-white transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
              >
                {excludeMutation.isPending ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    Excluding...
                  </>
                ) : (
                  <>
                    <UserMinus className="w-4 h-4" />
                    Exclude
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Header */}
      <div className="flex items-start justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-3">
            <Users className="w-7 h-7 text-slate-400" />
            Players
          </h1>
          <p className="text-slate-500 text-sm mt-1">
            Manage and monitor {players?.length || 0} team players
          </p>
        </div>

        {/* Excluded Players Toggle */}
        {(excludedPlayers?.length ?? 0) > 0 && (
          <button
            onClick={() => setShowExcluded(!showExcluded)}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all ${
              showExcluded 
                ? 'bg-orange-500/20 text-orange-400 border border-orange-500/30' 
                : 'bg-slate-800/50 text-slate-400 border border-slate-700/50 hover:bg-slate-800'
            }`}
          >
            {showExcluded ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
            Excluded ({excludedPlayers?.length})
          </button>
        )}
      </div>

      {/* Excluded Players Panel */}
      {showExcluded && excludedPlayers && excludedPlayers.length > 0 && (
        <div className="card p-5 border-orange-500/30 bg-orange-500/5 animate-slide-in-up">
          <div className="flex items-center gap-2 mb-4">
            <EyeOff className="w-5 h-5 text-orange-400" />
            <h3 className="font-semibold text-white">Excluded Players</h3>
            <span className="text-xs text-slate-500">Click to restore</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {excludedPlayers.map((name) => (
              <button
                key={name}
                onClick={() => restoreMutation.mutate(name)}
                disabled={restoreMutation.isPending}
                className="flex items-center gap-2 px-3 py-2 bg-slate-800/50 hover:bg-slate-800 border border-slate-700/50 rounded-lg text-sm text-slate-300 hover:text-white transition-all group"
              >
                <span>{name}</span>
                <UserPlus className="w-4 h-4 text-emerald-400 opacity-0 group-hover:opacity-100 transition-opacity" />
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Stats Overview */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="card p-4">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-slate-800/50 rounded-lg border border-slate-700/50">
              <Users className="w-5 h-5 text-slate-400" />
            </div>
            <div>
              <p className="text-2xl font-bold text-white">{riskCounts.all}</p>
              <p className="text-xs text-slate-500">Active Players</p>
            </div>
          </div>
        </div>
        <div className="card p-4">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-emerald-500/10 rounded-lg">
              <Activity className="w-5 h-5 text-emerald-400" />
            </div>
            <div>
              <p className="text-2xl font-bold text-emerald-400">{riskCounts.low}</p>
              <p className="text-xs text-slate-500">Low Risk</p>
            </div>
          </div>
        </div>
        <div className="card p-4">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-yellow-500/10 rounded-lg">
              <AlertTriangle className="w-5 h-5 text-yellow-400" />
            </div>
            <div>
              <p className="text-2xl font-bold text-yellow-400">{riskCounts.medium}</p>
              <p className="text-xs text-slate-500">Medium Risk</p>
            </div>
          </div>
        </div>
        <div className="card p-4">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-red-500/10 rounded-lg">
              <AlertTriangle className="w-5 h-5 text-red-400" />
            </div>
            <div>
              <p className="text-2xl font-bold text-red-400">{riskCounts.high}</p>
              <p className="text-xs text-slate-500">High Risk</p>
            </div>
          </div>
        </div>
      </div>

      {/* Load Distribution Chart */}
      {loadChartData.length > 0 && (
        <div className="card p-6">
          <div className="flex items-center gap-2 mb-4">
            <BarChart3 className="w-5 h-5 text-slate-400" />
            <h3 className="font-semibold text-white">Top 10 Players by Load</h3>
          </div>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={loadChartData} layout="vertical">
                <XAxis type="number" stroke="#475569" fontSize={11} />
                <YAxis dataKey="name" type="category" stroke="#475569" fontSize={11} width={70} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(15, 23, 42, 0.95)',
                    border: '1px solid rgba(51, 65, 85, 0.5)',
                    borderRadius: '12px',
                  }}
                  formatter={(value) => [`${value} load`, 'Avg Load']}
                />
                <Bar dataKey="load" radius={[0, 4, 4, 0]}>
                  {loadChartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={riskColors[entry.risk as keyof typeof riskColors].color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-4">
        <div className="flex-1 relative">
          <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-500" />
          <input
            type="text"
            placeholder="Search players..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full pl-11 pr-4 py-3 bg-slate-800/50 border border-slate-700/50 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500 transition-colors"
          />
        </div>
        
        <div className="flex gap-2">
          {(['all', 'low', 'medium', 'high'] as const).map((filter) => (
            <button
              key={filter}
              onClick={() => setRiskFilter(filter)}
              className={`px-4 py-2.5 rounded-xl font-medium text-sm transition-all ${
                riskFilter === filter
                  ? filter === 'all' ? 'btn-primary text-white' :
                    filter === 'high' ? 'bg-red-500 text-white shadow-lg shadow-red-500/25' :
                    filter === 'medium' ? 'bg-yellow-500 text-slate-900 shadow-lg shadow-yellow-500/25' :
                    'bg-emerald-500 text-white shadow-lg shadow-emerald-500/25'
                  : 'bg-slate-800/50 border border-slate-700/50 text-slate-400 hover:bg-slate-800 hover:text-white'
              }`}
            >
              {filter.charAt(0).toUpperCase() + filter.slice(1)} ({riskCounts[filter]})
            </button>
          ))}
        </div>
      </div>

      {/* Players Grid */}
      {isLoading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="card p-5 animate-pulse">
              <div className="flex items-center gap-4 mb-4">
                <div className="w-14 h-14 bg-slate-800 rounded-xl" />
                <div className="flex-1">
                  <div className="h-5 bg-slate-800 rounded w-32 mb-2" />
                  <div className="h-4 bg-slate-800 rounded w-20" />
                </div>
              </div>
              <div className="grid grid-cols-3 gap-3">
                <div className="h-20 bg-slate-800 rounded-lg" />
                <div className="h-20 bg-slate-800 rounded-lg" />
                <div className="h-20 bg-slate-800 rounded-lg" />
              </div>
            </div>
          ))}
        </div>
      ) : filteredPlayers.length === 0 ? (
        <div className="card p-12 text-center">
          <div className="w-16 h-16 mx-auto mb-4 bg-slate-800/50 rounded-2xl flex items-center justify-center">
            <Users className="w-8 h-8 text-slate-600" />
          </div>
          <h3 className="text-lg font-semibold text-white mb-2">No Players Found</h3>
          <p className="text-slate-500 text-sm">
            {search ? `No players match "${search}"` : 'No players in this category'}
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
          {filteredPlayers.map((player) => (
            <PlayerCard key={player.id} player={player} onExclude={handleExcludeClick} />
          ))}
        </div>
      )}
    </div>
  );
}
