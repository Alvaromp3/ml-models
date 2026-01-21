import { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { 
  BarChart3, 
  AlertTriangle, 
  TrendingUp, 
  Loader2, 
  CheckCircle, 
  Activity,
  Shield,
  Zap,
  Target,
  Heart,
  ArrowRight,
  Info,
  Bot,
  Sparkles,
  RefreshCw,
  Clock
} from 'lucide-react';
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import { playersApi, analysisApi, useDataStatus } from '../services/api';
import type { RiskPrediction, Player } from '../types';
import ReactMarkdown from 'react-markdown';

export default function Analysis() {
  const [selectedPlayer, setSelectedPlayer] = useState<string>('');
  const [prediction, setPrediction] = useState<RiskPrediction | null>(null);
  const [selectedPlayerData, setSelectedPlayerData] = useState<Player | null>(null);
  const [aiRecommendations, setAiRecommendations] = useState<any>(null);

  const { data: dataStatus } = useDataStatus();

  const { data: players } = useQuery({
    queryKey: ['players'],
    queryFn: playersApi.getAll,
    enabled: !!dataStatus?.loaded,
  });

  // Ollama status query
  const { data: ollamaStatus } = useQuery({
    queryKey: ['ollama', 'status'],
    queryFn: analysisApi.getOllamaStatus,
    staleTime: 30000,
  });

  const predictMutation = useMutation({
    mutationFn: (playerId: string) => analysisApi.predictRisk(playerId),
    onSuccess: (data) => {
      setPrediction(data);
      setAiRecommendations(null);
    },
    onError: (error: Error) => {
      console.error('Prediction error:', error);
    }
  });

  const aiMutation = useMutation({
    mutationFn: (playerId: string) => analysisApi.getAIRecommendations(playerId),
    onSuccess: (data) => setAiRecommendations(data),
  });

  const handlePlayerSelect = (playerId: string) => {
    setSelectedPlayer(playerId);
    const player = players?.find(p => p.id === playerId);
    setSelectedPlayerData(player || null);
    setPrediction(null);
    setAiRecommendations(null);
  };

  const handlePredict = () => {
    if (selectedPlayer) {
      predictMutation.mutate(selectedPlayer);
    }
  };

  const handleGetAIRecommendations = () => {
    if (selectedPlayer) {
      aiMutation.mutate(selectedPlayer);
    }
  };

  const riskConfig = {
    low: { 
      bg: 'bg-emerald-500/15', 
      text: 'text-emerald-400', 
      border: 'border-emerald-500/30',
      icon: CheckCircle,
      color: '#22c55e',
      gradient: 'from-emerald-500 to-green-600'
    },
    medium: { 
      bg: 'bg-yellow-500/15', 
      text: 'text-yellow-400', 
      border: 'border-yellow-500/30',
      icon: AlertTriangle,
      color: '#eab308',
      gradient: 'from-yellow-500 to-orange-500'
    },
    high: { 
      bg: 'bg-red-500/15', 
      text: 'text-red-400', 
      border: 'border-red-500/30',
      icon: AlertTriangle,
      color: '#ef4444',
      gradient: 'from-red-500 to-rose-600'
    },
  };

  // No data loaded state
  if (!dataStatus?.loaded) {
    return (
      <div className="min-h-[70vh] flex items-center justify-center animate-fade-in">
        <div className="card p-8 max-w-md text-center">
          <div className="w-16 h-16 mx-auto mb-6 bg-gradient-to-br from-orange-500 to-red-600 rounded-2xl flex items-center justify-center shadow-lg shadow-orange-500/30">
            <AlertTriangle className="w-8 h-8 text-white" />
          </div>
          <h2 className="text-xl font-bold text-white mb-2">No Data Loaded</h2>
          <p className="text-slate-400 text-sm mb-6">
            Load CSV data first to analyze player risk. Go to Dashboard to upload your data.
          </p>
          <a 
            href="/"
            className="inline-flex items-center gap-2 px-5 py-2.5 btn-primary rounded-xl font-medium text-white text-sm"
          >
            Go to Dashboard
            <ArrowRight className="w-4 h-4" />
          </a>
        </div>
      </div>
    );
  }

  // Generate radar chart data from player
  const getRadarData = () => {
    if (!selectedPlayerData) return [];
    return [
      { metric: 'Load', value: Math.min(100, (selectedPlayerData.avgLoad / 600) * 100), fullMark: 100 },
      { metric: 'Speed', value: Math.min(100, (selectedPlayerData.avgSpeed / 25) * 100), fullMark: 100 },
      { metric: 'Sessions', value: Math.min(100, (selectedPlayerData.sessions / 30) * 100), fullMark: 100 },
      { metric: 'Consistency', value: 75, fullMark: 100 },
      { metric: 'Recovery', value: selectedPlayerData.riskLevel === 'low' ? 90 : selectedPlayerData.riskLevel === 'medium' ? 60 : 30, fullMark: 100 },
    ];
  };

  // Risk probability chart data
  const getProbabilityData = () => {
    if (!prediction) return [];
    const prob = prediction.probability * 100;
    return [
      { name: 'Risk', value: prob, color: riskConfig[prediction.riskLevel].color },
      { name: 'Safe', value: 100 - prob, color: '#1e293b' },
    ];
  };

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white flex items-center gap-3">
          <Shield className="w-7 h-7 text-slate-400" />
          Risk Analysis
        </h1>
        <p className="text-slate-500 text-sm mt-1">
          Predict injury risk and get AI-powered recommendations
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Panel - Player Selection */}
        <div className="lg:col-span-1 space-y-4">
          {/* Select Player Card */}
          <div className="card p-6">
            <div className="flex items-center gap-3 mb-5">
              <div className="p-2.5 rounded-xl bg-slate-800/60 border border-slate-700/50">
                <Target className="w-5 h-5 text-slate-300" />
              </div>
              <div>
                <h2 className="font-semibold text-white">Select Player</h2>
                <p className="text-xs text-slate-500">Choose a player to analyze</p>
              </div>
            </div>

            <div className="space-y-4">
              <select
                value={selectedPlayer}
                onChange={(e) => handlePlayerSelect(e.target.value)}
                className="w-full px-4 py-3 bg-slate-800/50 border border-slate-700/50 rounded-xl text-white focus:outline-none focus:border-cyan-500 transition-colors"
              >
                <option value="">Choose a player...</option>
                {players?.map((player) => (
                  <option key={player.id} value={player.id}>
                    {player.name}
                  </option>
                ))}
              </select>

              <button
                onClick={handlePredict}
                disabled={!selectedPlayer || predictMutation.isPending}
                className="w-full py-3 btn-primary rounded-xl font-semibold text-white flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {predictMutation.isPending ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Activity className="w-5 h-5" />
                    Analyze Risk
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Selected Player Info */}
          {selectedPlayerData && (
            <div className="card p-6 animate-slide-in-up">
              <div className="flex items-center gap-4 mb-4">
                <div className={`
                  w-14 h-14 rounded-xl flex items-center justify-center font-bold text-xl
                  ${riskConfig[selectedPlayerData.riskLevel].bg} ${riskConfig[selectedPlayerData.riskLevel].text}
                  border ${riskConfig[selectedPlayerData.riskLevel].border}
                `}>
                  {selectedPlayerData.number}
                </div>
                <div>
                  <h3 className="font-semibold text-white">{selectedPlayerData.name}</h3>
                  <p className="text-sm text-slate-500">#{selectedPlayerData.number}</p>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="p-3 bg-slate-800/50 rounded-lg text-center">
                  <Zap className="w-4 h-4 text-slate-400 mx-auto mb-1" />
                  <p className="text-lg font-bold text-white">{selectedPlayerData.avgLoad}</p>
                  <p className="text-[10px] text-slate-500">Avg Load</p>
                </div>
                <div className="p-3 bg-slate-800/50 rounded-lg text-center">
                  <TrendingUp className="w-4 h-4 text-slate-400 mx-auto mb-1" />
                  <p className="text-lg font-bold text-white">{selectedPlayerData.avgSpeed}</p>
                  <p className="text-[10px] text-slate-500">Avg Speed</p>
                </div>
                <div className="p-3 bg-slate-800/50 rounded-lg text-center">
                  <Activity className="w-4 h-4 text-slate-400 mx-auto mb-1" />
                  <p className="text-lg font-bold text-white">{selectedPlayerData.sessions}</p>
                  <p className="text-[10px] text-slate-500">Sessions</p>
                </div>
                <div className="p-3 bg-slate-800/50 rounded-lg text-center">
                  <Heart className="w-4 h-4 text-red-400 mx-auto mb-1" />
                  <p className={`text-lg font-bold capitalize ${riskConfig[selectedPlayerData.riskLevel].text}`}>
                    {selectedPlayerData.riskLevel}
                  </p>
                  <p className="text-[10px] text-slate-500">Current Risk</p>
                </div>
              </div>
            </div>
          )}

          {/* Player Radar Chart */}
          {selectedPlayerData && (
            <div className="card p-6 animate-slide-in-up" style={{ animationDelay: '100ms' }}>
              <h3 className="font-semibold text-white mb-4 flex items-center gap-2">
                <BarChart3 className="w-4 h-4 text-slate-400" />
                Performance Profile
              </h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart data={getRadarData()}>
                    <PolarGrid stroke="#334155" />
                    <PolarAngleAxis dataKey="metric" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                    <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fill: '#64748b', fontSize: 10 }} />
                    <Radar
                      name="Performance"
                      dataKey="value"
                      stroke="#06b6d4"
                      fill="#06b6d4"
                      fillOpacity={0.3}
                      strokeWidth={2}
                    />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </div>

        {/* Right Panel - Results */}
        <div className="lg:col-span-2 space-y-4">
          {prediction ? (
            <>
              {/* No Recent Data Warning */}
              {prediction.hasRecentData === false && (
                <div className="card p-4 bg-yellow-500/10 border-yellow-500/30 animate-slide-in-up">
                  <div className="flex items-start gap-3">
                    <Clock className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" />
                    <div className="text-sm">
                      <p className="text-yellow-400 font-medium mb-1">No Recent Training Data</p>
                      <p className="text-slate-400">
                        This player has no training sessions in the last 45 days. 
                        Risk is automatically set to LOW as we cannot accurately assess without recent data.
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* Risk Level Card */}
              <div className={`card p-6 border ${riskConfig[prediction.riskLevel].border} animate-slide-in-up`}>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div className={`
                      w-16 h-16 rounded-2xl flex items-center justify-center
                      bg-gradient-to-br ${riskConfig[prediction.riskLevel].gradient}
                      shadow-lg
                    `}>
                      {(() => {
                        const IconComponent = riskConfig[prediction.riskLevel].icon;
                        return <IconComponent className="w-8 h-8 text-white" />;
                      })()}
                    </div>
                    <div>
                      <p className="text-sm text-slate-400 mb-1">Injury Risk Level</p>
                      <p className={`text-3xl font-bold capitalize ${riskConfig[prediction.riskLevel].text}`}>
                        {prediction.riskLevel} Risk
                      </p>
                      <p className="text-sm text-slate-500 mt-1">
                        {prediction.playerName}
                      </p>
                    </div>
                  </div>

                  {/* Probability Donut */}
                  <div className="w-32 h-32 relative">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={getProbabilityData()}
                          cx="50%"
                          cy="50%"
                          innerRadius={35}
                          outerRadius={50}
                          dataKey="value"
                          strokeWidth={0}
                        >
                          {getProbabilityData().map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.color} />
                          ))}
                        </Pie>
                      </PieChart>
                    </ResponsiveContainer>
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div className="text-center">
                        <p className={`text-2xl font-bold ${riskConfig[prediction.riskLevel].text}`}>
                          {(prediction.probability * 100).toFixed(0)}%
                        </p>
                        <p className="text-[10px] text-slate-500">Confidence</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Risk Factors & Recommendations */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Risk Factors */}
                <div className="card p-6 animate-slide-in-up" style={{ animationDelay: '100ms' }}>
                  <div className="flex items-center gap-2 mb-4">
                    <AlertTriangle className="w-5 h-5 text-orange-400" />
                    <h3 className="font-semibold text-white">Risk Factors</h3>
                  </div>
                  <ul className="space-y-3">
                    {prediction.factors.map((factor, i) => (
                      <li key={i} className="flex items-start gap-3 p-3 bg-slate-800/30 rounded-lg">
                        <span className={`
                          w-2 h-2 rounded-full mt-1.5 flex-shrink-0
                          ${prediction.riskLevel === 'high' ? 'bg-red-400' : prediction.riskLevel === 'medium' ? 'bg-yellow-400' : 'bg-emerald-400'}
                        `} />
                        <span className="text-sm text-slate-300">{factor}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Recommendations */}
                <div className="card p-6 animate-slide-in-up" style={{ animationDelay: '150ms' }}>
                  <div className="flex items-center gap-2 mb-4">
                    <CheckCircle className="w-5 h-5 text-emerald-400" />
                    <h3 className="font-semibold text-white">Recommendations</h3>
                  </div>
                  <ul className="space-y-3">
                    {prediction.recommendations.map((rec, i) => (
                      <li key={i} className="flex items-start gap-3 p-3 bg-emerald-500/5 border border-emerald-500/10 rounded-lg">
                        <CheckCircle className="w-4 h-4 text-emerald-400 mt-0.5 flex-shrink-0" />
                        <span className="text-sm text-slate-300">{rec}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>

              {/* AI Coach Section */}
              <div className="card p-6 animate-slide-in-up" style={{ animationDelay: '200ms' }}>
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-3">
                    <div className="p-2.5 rounded-xl bg-slate-800/60 border border-slate-700/50">
                      <Bot className="w-5 h-5 text-slate-300" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-white">AI Coach Recommendations</h3>
                      <p className="text-xs text-slate-500">
                        Powered by Ollama {ollamaStatus?.status === 'ready' ? '(Connected)' : '(Fallback mode)'}
                      </p>
                    </div>
                  </div>
                  
                  <button
                    onClick={handleGetAIRecommendations}
                    disabled={aiMutation.isPending}
                    className="px-4 py-2 bg-slate-800 hover:bg-slate-700 border border-slate-700/50 rounded-lg font-medium text-white text-sm flex items-center gap-2 disabled:opacity-50 transition-all"
                  >
                    {aiMutation.isPending ? (
                      <>
                        <RefreshCw className="w-4 h-4 animate-spin" />
                        Generating...
                      </>
                    ) : (
                      <>
                        <Sparkles className="w-4 h-4" />
                        Get AI Analysis
                      </>
                    )}
                  </button>
                </div>

                {/* Ollama Status Badge */}
                {ollamaStatus && (
                  <div className={`mb-4 px-3 py-2 rounded-lg text-xs inline-flex items-center gap-2 ${
                    ollamaStatus.status === 'ready' 
                      ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                      : 'bg-yellow-500/10 text-yellow-400 border border-yellow-500/20'
                  }`}>
                    <span className={`w-2 h-2 rounded-full ${ollamaStatus.status === 'ready' ? 'bg-emerald-400' : 'bg-yellow-400'}`} />
                    {ollamaStatus.status === 'ready' 
                      ? `Ollama connected - Model: ${ollamaStatus.defaultModel}`
                      : ollamaStatus.message}
                  </div>
                )}

                {/* AI Recommendations Content */}
                {aiRecommendations ? (
                    <div className="p-4 bg-slate-800/30 rounded-xl">
                    <div className="flex items-center gap-2 mb-3">
                      <Bot className="w-4 h-4 text-slate-400" />
                      <span className="text-sm font-medium text-slate-300">
                        {aiRecommendations.aiSuccess ? 'AI Analysis' : 'Standard Recommendations'}
                      </span>
                      {!aiRecommendations.aiSuccess && aiRecommendations.aiError && (
                        <span className="text-xs text-slate-500">
                          ({aiRecommendations.aiError})
                        </span>
                      )}
                    </div>
                    <div className="prose prose-invert prose-sm max-w-none">
                      <ReactMarkdown
                        components={{
                          h2: ({children}) => <h2 className="text-lg font-semibold text-white mt-4 mb-2">{children}</h2>,
                          h3: ({children}) => <h3 className="text-md font-medium text-slate-300 mt-3 mb-1">{children}</h3>,
                          p: ({children}) => <p className="text-sm text-slate-400 mb-2">{children}</p>,
                          ul: ({children}) => <ul className="list-disc list-inside text-sm text-slate-400 space-y-1">{children}</ul>,
                          ol: ({children}) => <ol className="list-decimal list-inside text-sm text-slate-400 space-y-1">{children}</ol>,
                          li: ({children}) => <li className="text-slate-400">{children}</li>,
                          strong: ({children}) => <strong className="text-white font-semibold">{children}</strong>,
                        }}
                      >
                        {aiRecommendations.aiRecommendations}
                      </ReactMarkdown>
                    </div>
                  </div>
                ) : (
                  <div className="p-6 bg-slate-800/20 rounded-xl text-center">
                    <Bot className="w-10 h-10 text-slate-600 mx-auto mb-3" />
                    <p className="text-sm text-slate-500">
                      Click "Get AI Analysis" to receive personalized coaching recommendations based on this player's performance data.
                    </p>
                  </div>
                )}
              </div>

              {/* Info Box */}
              <div className="card p-4 bg-slate-800/30 border-slate-700/50 animate-slide-in-up" style={{ animationDelay: '250ms' }}>
                <div className="flex items-start gap-3">
                  <Info className="w-5 h-5 text-slate-400 flex-shrink-0 mt-0.5" />
                  <div className="text-sm text-slate-400">
                    <p className="text-slate-300 font-medium mb-1">About this prediction</p>
                    <p>
                      Risk analysis uses data from the <strong>last 45 days only</strong>. If no recent training data is available, 
                      risk is set to LOW. AI recommendations are powered by Ollama (when available) or use rule-based fallback.
                    </p>
                  </div>
                </div>
              </div>
            </>
          ) : predictMutation.isError ? (
            <div className="card p-8 text-center animate-slide-in-up">
              <div className="w-16 h-16 mx-auto mb-4 bg-red-500/10 rounded-2xl flex items-center justify-center">
                <AlertTriangle className="w-8 h-8 text-red-400" />
              </div>
              <h3 className="text-lg font-semibold text-white mb-2">Analysis Failed</h3>
              <p className="text-slate-400 text-sm mb-4">
                {(predictMutation.error as Error)?.message || 'Could not analyze player risk. Please try again.'}
              </p>
              <button
                onClick={handlePredict}
                className="px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg text-sm text-white transition-colors"
              >
                Try Again
              </button>
            </div>
          ) : (
            <div className="card p-12 text-center">
              <div className="w-20 h-20 mx-auto mb-6 bg-slate-800/50 rounded-2xl flex items-center justify-center">
                <Shield className="w-10 h-10 text-slate-600" />
              </div>
              <h3 className="text-lg font-semibold text-white mb-2">Ready to Analyze</h3>
              <p className="text-slate-500 text-sm max-w-md mx-auto">
                Select a player from the list and click "Analyze Risk" to get a comprehensive injury risk assessment with AI-powered recommendations.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
