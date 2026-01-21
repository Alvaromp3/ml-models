import { useState, useMemo } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { 
  Users, 
  Zap, 
  TrendingUp, 
  Shield, 
  Target,
  ChevronRight,
  Activity,
  Gauge,
  Trophy,
  BarChart3,
  Sparkles,
  Calculator,
  Play,
  Clock,
  RefreshCw,
  Star
} from 'lucide-react';
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
  Legend,
  PieChart,
  Pie,
  LineChart,
  Line,
  CartesianGrid,
} from 'recharts';
import { playersApi, trainingApi, useDataStatus } from '../services/api';
import type { Player } from '../types';

type CriteriaType = 'balanced' | 'speed' | 'load' | 'lowRisk' | 'highIntensity';

// Simple layout positions for 11 players (no position labels)
const fieldLayout = [
  { x: 50, y: 88 },  // GK
  { x: 18, y: 68 },  // DEF
  { x: 38, y: 72 },
  { x: 62, y: 72 },
  { x: 82, y: 68 },
  { x: 28, y: 48 },  // MID
  { x: 50, y: 42 },
  { x: 72, y: 48 },
  { x: 20, y: 22 },  // FWD
  { x: 50, y: 15 },
  { x: 80, y: 22 },
];

const criteriaConfig: Record<CriteriaType, { label: string; description: string; icon: typeof Zap; color: string; gradient: string }> = {
  balanced: { label: 'Balanced', description: 'Best overall', icon: Trophy, color: 'cyan', gradient: 'from-cyan-500 to-blue-600' },
  speed: { label: 'Max Speed', description: 'Fastest players', icon: Gauge, color: 'emerald', gradient: 'from-emerald-500 to-green-600' },
  load: { label: 'High Load', description: 'Work capacity', icon: Zap, color: 'orange', gradient: 'from-orange-500 to-red-600' },
  lowRisk: { label: 'Low Risk', description: 'Safest players', icon: Shield, color: 'green', gradient: 'from-green-500 to-emerald-600' },
  highIntensity: { label: 'Intensity', description: 'Match ready', icon: Activity, color: 'red', gradient: 'from-red-500 to-rose-600' },
};

export default function Lineup() {
  const [selectedCriteria, setSelectedCriteria] = useState<CriteriaType>('balanced');
  const [activeTab, setActiveTab] = useState<'lineup' | 'comparison' | 'stats' | 'prediction'>('lineup');
  const [selectedPlayerForPrediction, setSelectedPlayerForPrediction] = useState<string>('');
  const [sessionType, setSessionType] = useState<'match' | 'training'>('match');

  const { data: dataStatus } = useDataStatus();

  const { data: players, isLoading } = useQuery({
    queryKey: ['players'],
    queryFn: playersApi.getAll,
    enabled: !!dataStatus?.loaded,
  });

  // Predict load mutation
  const predictLoadMutation = useMutation({
    mutationFn: async (playerId: string) => {
      const player = players?.find(p => p.id === playerId);
      if (!player) throw new Error('Player not found');
      
      const detail = await playersApi.getDetail(playerId);
      const result = await trainingApi.predictLoad({
        playerId,
        sessionType,
        features: detail.metrics
      });
      
      return { player, result };
    },
  });

  // Generate best lineup based on criteria
  const bestLineup = useMemo(() => {
    if (!players || players.length < 11) return [];

    let sortedPlayers = [...players];

    switch (selectedCriteria) {
      case 'speed':
        sortedPlayers.sort((a, b) => b.avgSpeed - a.avgSpeed);
        break;
      case 'load':
        sortedPlayers.sort((a, b) => b.avgLoad - a.avgLoad);
        break;
      case 'lowRisk':
        const riskOrder = { low: 0, medium: 1, high: 2 };
        sortedPlayers.sort((a, b) => {
          const riskDiff = riskOrder[a.riskLevel] - riskOrder[b.riskLevel];
          if (riskDiff !== 0) return riskDiff;
          return b.avgSpeed - a.avgSpeed;
        });
        break;
      case 'highIntensity':
        sortedPlayers.sort((a, b) => (b.avgLoad * b.avgSpeed) - (a.avgLoad * a.avgSpeed));
        break;
      case 'balanced':
      default:
        sortedPlayers.sort((a, b) => {
          const riskScore = { low: 1, medium: 0.7, high: 0.4 };
          const scoreA = (a.avgSpeed / 25) * 0.3 + (a.avgLoad / 600) * 0.3 + riskScore[a.riskLevel] * 0.4;
          const scoreB = (b.avgSpeed / 25) * 0.3 + (b.avgLoad / 600) * 0.3 + riskScore[b.riskLevel] * 0.4;
          return scoreB - scoreA;
        });
        break;
    }

    return sortedPlayers.slice(0, 11);
  }, [players, selectedCriteria]);

  // Stats for the lineup
  const lineupStats = useMemo(() => {
    if (!bestLineup.length) return null;
    
    const avgSpeed = bestLineup.reduce((sum, p) => sum + p.avgSpeed, 0) / bestLineup.length;
    const avgLoad = bestLineup.reduce((sum, p) => sum + p.avgLoad, 0) / bestLineup.length;
    const totalSessions = bestLineup.reduce((sum, p) => sum + p.sessions, 0);
    const riskCounts = {
      low: bestLineup.filter(p => p.riskLevel === 'low').length,
      medium: bestLineup.filter(p => p.riskLevel === 'medium').length,
      high: bestLineup.filter(p => p.riskLevel === 'high').length,
    };

    return { avgSpeed, avgLoad, totalSessions, riskCounts };
  }, [bestLineup]);

  // Radar data for lineup
  const radarData = useMemo(() => {
    if (!lineupStats) return [];
    return [
      { metric: 'Speed', value: (lineupStats.avgSpeed / 25) * 100, fullMark: 100 },
      { metric: 'Load', value: (lineupStats.avgLoad / 600) * 100, fullMark: 100 },
      { metric: 'Low Risk', value: (lineupStats.riskCounts.low / 11) * 100, fullMark: 100 },
      { metric: 'Experience', value: Math.min(100, (lineupStats.totalSessions / 200) * 100), fullMark: 100 },
      { metric: 'Fitness', value: 100 - (lineupStats.riskCounts.high / 11) * 100, fullMark: 100 },
    ];
  }, [lineupStats]);

  // Bar chart data for player comparison
  const comparisonData = useMemo(() => {
    return bestLineup.map(p => ({
      name: p.name.split(' ')[0],
      speed: p.avgSpeed,
      load: p.avgLoad / 10,
      risk: p.riskLevel === 'low' ? 100 : p.riskLevel === 'medium' ? 60 : 30,
    }));
  }, [bestLineup]);

  // Risk distribution pie chart
  const riskPieData = useMemo(() => {
    if (!lineupStats) return [];
    return [
      { name: 'Low Risk', value: lineupStats.riskCounts.low, color: '#22c55e' },
      { name: 'Medium Risk', value: lineupStats.riskCounts.medium, color: '#eab308' },
      { name: 'High Risk', value: lineupStats.riskCounts.high, color: '#ef4444' },
    ].filter(d => d.value > 0);
  }, [lineupStats]);

  // No data state
  if (!dataStatus?.loaded) {
    return (
      <div className="min-h-[70vh] flex items-center justify-center animate-fade-in">
        <div className="card p-8 max-w-md text-center">
          <div className="w-16 h-16 mx-auto mb-6 bg-slate-800/60 border border-slate-700/50 rounded-2xl flex items-center justify-center">
            <Trophy className="w-8 h-8 text-slate-300" />
          </div>
          <h2 className="text-xl font-bold text-white mb-2">No Data Loaded</h2>
          <p className="text-slate-400 text-sm mb-6">
            Load CSV data first to generate lineups.
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

  if (isLoading) {
    return (
      <div className="min-h-[70vh] flex items-center justify-center">
        <div className="animate-spin w-8 h-8 border-2 border-cyan-500 border-t-transparent rounded-full" />
      </div>
    );
  }

  const config = criteriaConfig[selectedCriteria];

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex items-start justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-3">
            <div className="p-2 bg-slate-800/60 border border-slate-700/50 rounded-xl">
              <Trophy className="w-5 h-5 text-slate-300" />
            </div>
            Best Lineup Generator
          </h1>
          <p className="text-slate-500 text-sm mt-2 ml-12">
            Generate optimal lineups and predict player load
          </p>
        </div>
        
        {/* Current criteria badge */}
        <div className="px-4 py-2 bg-slate-800/50 border border-slate-700/50 rounded-xl flex items-center gap-2">
          <config.icon className="w-4 h-4 text-slate-400" />
          <span className="text-sm font-medium text-slate-300">{config.label} Mode</span>
        </div>
      </div>

      {/* Criteria Selection - Horizontal scroll on mobile */}
      <div className="flex gap-3 overflow-x-auto pb-2 -mx-2 px-2">
        {(Object.entries(criteriaConfig) as [CriteriaType, typeof criteriaConfig.balanced][]).map(([key, cfg]) => {
          const Icon = cfg.icon;
          const isSelected = selectedCriteria === key;
          return (
            <button
              key={key}
              onClick={() => setSelectedCriteria(key)}
              className={`flex-shrink-0 p-4 rounded-xl border transition-all min-w-[140px] ${
                isSelected
                  ? 'bg-slate-800/60 border-slate-600 shadow-lg'
                  : 'bg-slate-800/30 border-slate-700/50 hover:bg-slate-800/50 hover:border-slate-600'
              }`}
            >
              <Icon className={`w-6 h-6 mx-auto mb-2 ${isSelected ? 'text-white' : 'text-slate-400'}`} />
              <p className={`text-sm font-medium text-center ${isSelected ? 'text-white' : 'text-slate-300'}`}>{cfg.label}</p>
              <p className={`text-[10px] text-center mt-1 ${isSelected ? 'text-white/70' : 'text-slate-500'}`}>{cfg.description}</p>
            </button>
          );
        })}
      </div>

      {/* Tabs */}
      <div className="flex gap-2 border-b border-slate-800 pb-2 overflow-x-auto">
        {[
          { id: 'lineup', label: 'Lineup', icon: Users },
          { id: 'comparison', label: 'Compare', icon: BarChart3 },
          { id: 'stats', label: 'Stats', icon: Activity },
          { id: 'prediction', label: 'Predict Load', icon: Calculator },
        ].map((tab) => {
          const Icon = tab.icon;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as typeof activeTab)}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium transition-all whitespace-nowrap ${
                activeTab === tab.id
                  ? 'bg-slate-800/60 border border-slate-700/50 text-white'
                  : 'text-slate-400 hover:text-white hover:bg-slate-800/50'
              }`}
            >
              <Icon className="w-4 h-4" />
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* Tab Content */}
      {activeTab === 'lineup' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Soccer Field */}
          <div className="lg:col-span-2 card p-6 relative overflow-hidden">
            {/* Background decoration */}
            <div className="absolute top-0 right-0 w-64 h-64 bg-slate-800/20 rounded-full blur-3xl" />
            
            <h3 className="font-semibold text-white mb-4 flex items-center gap-2 relative">
              <Star className="w-5 h-5 text-slate-400" />
              Best XI - {config.label}
            </h3>
            
            {/* Field */}
            <div className="relative w-full aspect-[4/5] max-h-[500px] bg-gradient-to-b from-emerald-900/50 via-emerald-800/40 to-emerald-900/50 rounded-2xl border border-emerald-700/30 overflow-hidden">
              {/* Field pattern */}
              <div className="absolute inset-0 opacity-20">
                <div className="absolute top-1/4 left-0 right-0 h-px bg-white/30" />
                <div className="absolute top-3/4 left-0 right-0 h-px bg-white/30" />
              </div>
              
              {/* Field markings */}
              <div className="absolute inset-4 border-2 border-white/25 rounded-lg" />
              <div className="absolute top-4 left-1/2 -translate-x-1/2 w-28 h-14 border-2 border-white/25 border-t-0 rounded-b-lg" />
              <div className="absolute bottom-4 left-1/2 -translate-x-1/2 w-28 h-14 border-2 border-white/25 border-b-0 rounded-t-lg" />
              <div className="absolute top-1/2 left-4 right-4 h-0.5 bg-white/20" />
              <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-20 h-20 border-2 border-white/25 rounded-full" />
              <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-2 h-2 bg-white/40 rounded-full" />

              {/* Players */}
              {fieldLayout.map((pos, idx) => {
                const player = bestLineup[idx];
                if (!player) return null;
                
                const riskColors = {
                  low: 'bg-emerald-500 shadow-emerald-500/50',
                  medium: 'bg-yellow-500 shadow-yellow-500/50',
                  high: 'bg-red-500 shadow-red-500/50',
                };
                
                return (
                  <div
                    key={idx}
                    className="absolute transform -translate-x-1/2 -translate-y-1/2 group cursor-pointer"
                    style={{ left: `${pos.x}%`, top: `${pos.y}%` }}
                  >
                    {/* Glow effect */}
                    <div className={`absolute inset-0 ${riskColors[player.riskLevel]} rounded-full blur-md opacity-50 group-hover:opacity-75 transition-opacity`} />
                    
                    {/* Player circle */}
                    <div className={`relative w-11 h-11 rounded-full ${riskColors[player.riskLevel]} flex items-center justify-center text-white font-bold text-sm shadow-lg border-2 border-white/60 transition-transform group-hover:scale-110`}>
                      {player.number}
                    </div>
                    
                    {/* Tooltip */}
                    <div className="absolute top-full left-1/2 -translate-x-1/2 mt-2 bg-slate-900/95 backdrop-blur px-3 py-2 rounded-lg text-center whitespace-nowrap opacity-0 group-hover:opacity-100 transition-all z-10 shadow-xl border border-slate-700/50">
                      <p className="text-xs font-semibold text-white">{player.name}</p>
                      <p className="text-[10px] text-slate-400">{player.avgSpeed} mph • {player.avgLoad} load</p>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Player List */}
          <div className="card p-6">
            <h3 className="font-semibold text-white mb-4 flex items-center gap-2">
              <Users className="w-4 h-4 text-slate-400" />
              Selected Players
            </h3>
            <div className="space-y-2 max-h-[500px] overflow-y-auto pr-2 custom-scrollbar">
              {bestLineup.map((player, idx) => {
                const riskColors = {
                  low: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30',
                  medium: 'text-yellow-400 bg-yellow-500/10 border-yellow-500/30',
                  high: 'text-red-400 bg-red-500/10 border-red-500/30',
                };
                
                return (
                  <div 
                    key={player.id} 
                    className="flex items-center gap-3 p-3 bg-slate-800/30 hover:bg-slate-800/50 rounded-xl transition-all border border-transparent hover:border-slate-700/50"
                  >
                    <div className="w-9 h-9 rounded-lg bg-slate-800/60 border border-slate-700/50 text-white flex items-center justify-center text-sm font-bold">
                      {idx + 1}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-white truncate">{player.name}</p>
                      <div className="flex items-center gap-2 mt-0.5">
                        <span className="text-[10px] text-slate-500">#{player.number}</span>
                        <span className="text-[10px] text-slate-600">•</span>
                        <span className="text-[10px] text-slate-400">{player.avgSpeed} mph</span>
                      </div>
                    </div>
                    <span className={`text-[10px] px-2 py-1 rounded-full border ${riskColors[player.riskLevel]}`}>
                      {player.riskLevel}
                    </span>
                  </div>
                );
              })}
            </div>
            
            {/* Summary */}
            {lineupStats && (
              <div className="mt-4 pt-4 border-t border-slate-800 grid grid-cols-2 gap-3">
                <div className="text-center p-3 bg-slate-800/30 rounded-lg">
                  <p className="text-lg font-bold text-white">{lineupStats.avgSpeed.toFixed(1)}</p>
                  <p className="text-[10px] text-slate-500">Avg Speed</p>
                </div>
                <div className="text-center p-3 bg-slate-800/30 rounded-lg">
                  <p className="text-lg font-bold text-white">{lineupStats.avgLoad.toFixed(0)}</p>
                  <p className="text-[10px] text-slate-500">Avg Load</p>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {activeTab === 'comparison' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="card p-6">
            <h3 className="font-semibold text-white mb-4 flex items-center gap-2">
              <Gauge className="w-5 h-5 text-slate-400" />
              Speed Comparison
            </h3>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={comparisonData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
                  <XAxis type="number" stroke="#475569" fontSize={11} />
                  <YAxis dataKey="name" type="category" stroke="#475569" fontSize={10} width={60} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'rgba(15, 23, 42, 0.95)',
                      border: '1px solid rgba(51, 65, 85, 0.5)',
                      borderRadius: '12px',
                    }}
                  />
                  <Bar dataKey="speed" fill="url(#speedGradient)" radius={[0, 6, 6, 0]} name="Speed (mph)" />
                  <defs>
                    <linearGradient id="speedGradient" x1="0" y1="0" x2="1" y2="0">
                      <stop offset="0%" stopColor="#06b6d4" />
                      <stop offset="100%" stopColor="#3b82f6" />
                    </linearGradient>
                  </defs>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="card p-6">
            <h3 className="font-semibold text-white mb-4 flex items-center gap-2">
              <Zap className="w-5 h-5 text-orange-400" />
              Load Comparison
            </h3>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={comparisonData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
                  <XAxis type="number" stroke="#475569" fontSize={11} />
                  <YAxis dataKey="name" type="category" stroke="#475569" fontSize={10} width={60} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'rgba(15, 23, 42, 0.95)',
                      border: '1px solid rgba(51, 65, 85, 0.5)',
                      borderRadius: '12px',
                    }}
                    formatter={(value: number) => [`${(value * 10).toFixed(0)}`, 'Load']}
                  />
                  <Bar dataKey="load" fill="url(#loadGradient)" radius={[0, 6, 6, 0]} name="Load" />
                  <defs>
                    <linearGradient id="loadGradient" x1="0" y1="0" x2="1" y2="0">
                      <stop offset="0%" stopColor="#f97316" />
                      <stop offset="100%" stopColor="#ef4444" />
                    </linearGradient>
                  </defs>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="card p-6 lg:col-span-2">
            <h3 className="font-semibold text-white mb-4 flex items-center gap-2">
              <Activity className="w-5 h-5 text-emerald-400" />
              Performance Trend
            </h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={comparisonData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="name" stroke="#475569" fontSize={11} />
                  <YAxis stroke="#475569" fontSize={11} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'rgba(15, 23, 42, 0.95)',
                      border: '1px solid rgba(51, 65, 85, 0.5)',
                      borderRadius: '12px',
                    }}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="speed" stroke="#06b6d4" strokeWidth={3} dot={{ fill: '#06b6d4', strokeWidth: 2 }} name="Speed" />
                  <Line type="monotone" dataKey="risk" stroke="#22c55e" strokeWidth={3} dot={{ fill: '#22c55e', strokeWidth: 2 }} name="Fitness %" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'stats' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="card p-6">
            <h3 className="font-semibold text-white mb-4 flex items-center gap-2">
              <Target className="w-5 h-5 text-purple-400" />
              Team Profile
            </h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart data={radarData}>
                  <PolarGrid stroke="#334155" />
                  <PolarAngleAxis dataKey="metric" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                  <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fill: '#64748b', fontSize: 10 }} />
                  <Radar
                    name="Team"
                    dataKey="value"
                    stroke="#8b5cf6"
                    fill="url(#radarGradient)"
                    strokeWidth={2}
                  />
                  <defs>
                    <linearGradient id="radarGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#8b5cf6" stopOpacity={0.5} />
                      <stop offset="100%" stopColor="#8b5cf6" stopOpacity={0.1} />
                    </linearGradient>
                  </defs>
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="card p-6">
            <h3 className="font-semibold text-white mb-4 flex items-center gap-2">
              <Shield className="w-5 h-5 text-emerald-400" />
              Risk Distribution
            </h3>
            <div className="h-64 relative">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={riskPieData}
                    cx="50%"
                    cy="50%"
                    innerRadius={50}
                    outerRadius={75}
                    dataKey="value"
                    strokeWidth={0}
                    paddingAngle={4}
                  >
                    {riskPieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'rgba(15, 23, 42, 0.95)',
                      border: '1px solid rgba(51, 65, 85, 0.5)',
                      borderRadius: '12px',
                    }}
                    formatter={(value) => [`${value} players`, '']}
                  />
                </PieChart>
              </ResponsiveContainer>
              <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                <div className="text-center">
                  <p className="text-3xl font-bold text-white">11</p>
                  <p className="text-[10px] text-slate-500">Players</p>
                </div>
              </div>
            </div>
            <div className="flex justify-center gap-4 mt-2">
              {riskPieData.map((entry) => (
                <div key={entry.name} className="flex items-center gap-2">
                  <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: entry.color }} />
                  <span className="text-[10px] text-slate-400">{entry.value}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="card p-6">
            <h3 className="font-semibold text-white mb-4 flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-slate-400" />
              Team Summary
            </h3>
            <div className="space-y-4">
              {[
                { label: 'Avg Speed', value: lineupStats?.avgSpeed.toFixed(1), unit: 'mph', color: 'cyan', max: 25 },
                { label: 'Avg Load', value: lineupStats?.avgLoad.toFixed(0), unit: '', color: 'orange', max: 600 },
                { label: 'Team Fitness', value: lineupStats ? Math.round((lineupStats.riskCounts.low / 11) * 100) : 0, unit: '%', color: 'emerald', max: 100 },
              ].map((stat) => (
                <div key={stat.label} className="p-4 bg-slate-800/30 rounded-xl">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-slate-400 text-sm">{stat.label}</span>
                    <span className={`text-${stat.color}-400 font-bold`}>{stat.value}{stat.unit}</span>
                  </div>
                  <div className="h-2 bg-slate-700/50 rounded-full overflow-hidden">
                    <div 
                      className={`h-full bg-gradient-to-r from-${stat.color}-500 to-${stat.color}-400 rounded-full transition-all duration-500`}
                      style={{ width: `${(parseFloat(stat.value?.toString() || '0') / stat.max) * 100}%` }}
                    />
                  </div>
                </div>
              ))}
              
              <div className="p-4 bg-slate-800/30 rounded-xl">
                <div className="flex items-center justify-between">
                  <span className="text-slate-400 text-sm">Total Sessions</span>
                  <span className="text-purple-400 font-bold">{lineupStats?.totalSessions}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'prediction' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Prediction Form */}
          <div className="card p-6">
            <div className="flex items-center gap-3 mb-6">
              <div className="p-2.5 rounded-xl bg-slate-800/60 border border-slate-700/50">
                <Calculator className="w-5 h-5 text-slate-300" />
              </div>
              <div>
                <h3 className="font-semibold text-white">Predict Player Load</h3>
                <p className="text-sm text-slate-500">Calculate expected load for next session</p>
              </div>
            </div>

            {/* Session Type */}
            <div className="mb-5">
              <label className="block text-sm text-slate-400 mb-3">Session Type</label>
              <div className="grid grid-cols-2 gap-3">
                <button
                  onClick={() => setSessionType('match')}
                  className={`p-4 rounded-xl border transition-all flex items-center gap-3 ${
                    sessionType === 'match'
                      ? 'bg-gradient-to-r from-orange-500/20 to-red-500/20 border-orange-500/50'
                      : 'bg-slate-800/30 border-slate-700/50 hover:bg-slate-800/50'
                  }`}
                >
                  <Trophy className={`w-5 h-5 ${sessionType === 'match' ? 'text-orange-400' : 'text-slate-400'}`} />
                  <div className="text-left">
                    <p className={`text-sm font-medium ${sessionType === 'match' ? 'text-white' : 'text-slate-300'}`}>Match</p>
                    <p className="text-[10px] text-slate-500">+15% load factor</p>
                  </div>
                </button>
                <button
                  onClick={() => setSessionType('training')}
                  className={`p-4 rounded-xl border transition-all flex items-center gap-3 ${
                    sessionType === 'training'
                      ? 'bg-slate-800/50 border-slate-600/50'
                      : 'bg-slate-800/30 border-slate-700/50 hover:bg-slate-800/50'
                  }`}
                >
                  <Activity className={`w-5 h-5 ${sessionType === 'training' ? 'text-slate-300' : 'text-slate-400'}`} />
                  <div className="text-left">
                    <p className={`text-sm font-medium ${sessionType === 'training' ? 'text-white' : 'text-slate-300'}`}>Training</p>
                    <p className="text-[10px] text-slate-500">Standard load</p>
                  </div>
                </button>
              </div>
            </div>

            {/* Player Selection */}
            <div className="mb-5">
              <label className="block text-sm text-slate-400 mb-2">Select Player</label>
              <select
                value={selectedPlayerForPrediction}
                onChange={(e) => setSelectedPlayerForPrediction(e.target.value)}
                className="w-full p-3 bg-slate-800 border border-slate-700 rounded-xl text-white focus:outline-none focus:border-cyan-500 transition-colors"
              >
                <option value="">-- Select a player --</option>
                {players?.map((player) => (
                  <option key={player.id} value={player.id}>
                    {player.name} (#{player.number})
                  </option>
                ))}
              </select>
            </div>

            {/* Predict Button */}
            <button
              onClick={() => predictLoadMutation.mutate(selectedPlayerForPrediction)}
              disabled={!selectedPlayerForPrediction || predictLoadMutation.isPending}
              className="w-full py-3.5 bg-slate-800 hover:bg-slate-700 border border-slate-700/50 rounded-xl font-semibold text-white flex items-center justify-center gap-2 disabled:opacity-50 transition-all"
            >
              {predictLoadMutation.isPending ? (
                <>
                  <RefreshCw className="w-5 h-5 animate-spin" />
                  Calculating...
                </>
              ) : (
                <>
                  <Play className="w-5 h-5" />
                  Predict Load
                </>
              )}
            </button>
          </div>

          {/* Prediction Result */}
          <div className="card p-6">
            <h3 className="font-semibold text-white mb-4 flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-emerald-400" />
              Prediction Result
            </h3>

            {!predictLoadMutation.data && !predictLoadMutation.error && (
              <div className="h-64 flex items-center justify-center text-slate-500 text-sm">
                <div className="text-center">
                  <div className="w-16 h-16 mx-auto mb-4 bg-slate-800/50 rounded-2xl flex items-center justify-center">
                    <Calculator className="w-8 h-8 text-slate-600" />
                  </div>
                  <p>Select a player and click Predict</p>
                  <p className="text-[10px] text-slate-600 mt-1">to see expected load for next session</p>
                </div>
              </div>
            )}

            {predictLoadMutation.error && (
              <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-xl">
                <p className="text-sm text-red-400">Error: {(predictLoadMutation.error as Error).message}</p>
              </div>
            )}

            {predictLoadMutation.data && (
              <div className="space-y-4 animate-slide-in-up">
                {/* Player Info */}
                <div className="flex items-center gap-4 p-4 bg-slate-800/30 rounded-xl">
                  <div className="w-14 h-14 rounded-xl bg-slate-800/60 border border-slate-700/50 flex items-center justify-center text-white font-bold text-lg">
                    {predictLoadMutation.data.player.number}
                  </div>
                  <div>
                    <p className="font-semibold text-white">{predictLoadMutation.data.player.name}</p>
                    <p className="text-xs text-slate-500">{sessionType === 'match' ? 'Match' : 'Training'} Prediction</p>
                  </div>
                </div>

                {/* Predicted Load */}
                <div className="p-6 bg-slate-800/30 border border-slate-700/50 rounded-xl text-center relative overflow-hidden">
                  <p className="text-sm text-slate-400 mb-2 relative">Predicted Player Load</p>
                  <p className="text-6xl font-bold text-white relative">
                    {predictLoadMutation.data.result.predictedLoad?.toFixed(0) || 'N/A'}
                  </p>
                  <p className="text-xs text-slate-500 mt-2 relative">Based on ML model analysis</p>
                </div>

                {/* Additional Info */}
                <div className="grid grid-cols-2 gap-3">
                  <div className="p-3 bg-slate-800/30 rounded-lg text-center">
                    <p className="text-xs text-slate-500">Historical Avg</p>
                    <p className="text-lg font-bold text-white">{predictLoadMutation.data.player.avgLoad.toFixed(0)}</p>
                  </div>
                  <div className="p-3 bg-slate-800/30 rounded-lg text-center">
                    <p className="text-xs text-slate-500">Sessions</p>
                    <p className="text-lg font-bold text-white">{predictLoadMutation.data.player.sessions}</p>
                  </div>
                </div>

                {/* Note */}
                <div className="p-3 bg-slate-800/20 rounded-lg flex items-start gap-2">
                  <Clock className="w-4 h-4 text-slate-500 flex-shrink-0 mt-0.5" />
                  <p className="text-xs text-slate-500">
                    Prediction uses regression model trained on historical data.
                    {sessionType === 'match' && ' Match sessions include +15% intensity factor.'}
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
