import { useState, useRef } from 'react';
import { 
  Users, 
  Zap, 
  AlertTriangle, 
  Gauge, 
  Upload, 
  Database, 
  FileUp, 
  CheckCircle, 
  Activity,
  TrendingUp,
  Shield,
  Calendar,
  Clock,
  Sparkles
} from 'lucide-react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import KPICard from '../components/dashboard/KPICard';
import LoadChart from '../components/charts/LoadChart';
import RiskDonut from '../components/charts/RiskDonut';
import PlayerList from '../components/dashboard/PlayerList';
import {
  useDashboardKPIs,
  useLoadHistory,
  useRiskDistribution,
  useHighRiskPlayers,
  useTopPerformers,
  useDataStatus,
} from '../hooks/useDashboard';
import { dataApi } from '../services/api';

export default function Dashboard() {
  const queryClient = useQueryClient();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [uploadStatus, setUploadStatus] = useState<{ success: boolean; message: string } | null>(null);
  
  const { data: dataStatus } = useDataStatus();
  const { data: kpis } = useDashboardKPIs();
  const { data: loadHistory } = useLoadHistory();
  const { data: riskDistribution } = useRiskDistribution();
  const { data: highRiskPlayers } = useHighRiskPlayers();
  const { data: topPerformers } = useTopPerformers();

  const loadSampleMutation = useMutation({
    mutationFn: dataApi.loadSample,
    onSuccess: () => {
      queryClient.invalidateQueries();
      setUploadStatus({ success: true, message: 'Sample data loaded successfully!' });
    },
    onError: () => {
      setUploadStatus({ success: false, message: 'Error loading data. Make sure backend is running on port 8000.' });
    }
  });

  const uploadMutation = useMutation({
    mutationFn: dataApi.upload,
    onSuccess: (data) => {
      queryClient.invalidateQueries();
      setUploadStatus({ 
        success: true, 
        message: `CSV loaded: ${data.rowCount} rows, ${data.players.length} players` 
      });
    },
    onError: (error: Error) => {
      setUploadStatus({ success: false, message: `Upload error: ${error.message}` });
    }
  });

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (!file.name.endsWith('.csv')) {
        setUploadStatus({ success: false, message: 'Please select a CSV file' });
        return;
      }
      setUploadStatus(null);
      uploadMutation.mutate(file);
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  // Welcome screen when no data loaded
  if (!dataStatus?.loaded) {
    return (
      <div className="min-h-[85vh] flex items-center justify-center">
        <div className="max-w-lg w-full animate-fade-in">
          <div className="card p-8 text-center relative overflow-hidden">
            {/* Background decoration */}
            <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-cyan-500 via-blue-500 to-purple-500" />
            <div className="absolute -top-24 -right-24 w-48 h-48 bg-cyan-500/10 rounded-full blur-3xl" />
            <div className="absolute -bottom-24 -left-24 w-48 h-48 bg-purple-500/10 rounded-full blur-3xl" />
            
            {/* Icon */}
            <div className="relative w-20 h-20 mx-auto mb-6">
              <div className="absolute inset-0 bg-slate-800/50 rounded-2xl rotate-6 opacity-50" />
              <div className="relative w-full h-full bg-slate-800/60 border border-slate-700/50 rounded-2xl flex items-center justify-center">
                <Activity className="w-10 h-10 text-slate-300" />
              </div>
            </div>

            <h1 className="text-3xl font-bold text-white mb-3">
              Graceland Soccer Analytics
            </h1>
            <p className="text-slate-400 text-sm mb-8 leading-relaxed max-w-sm mx-auto">
              Upload your Catapult GPS data to analyze player performance, predict injury risks, and get AI-powered recommendations.
            </p>

            {/* Status message */}
            {uploadStatus && (
              <div className={`mb-6 p-4 rounded-xl flex items-center gap-3 text-sm ${
                uploadStatus.success 
                  ? 'bg-emerald-500/10 border border-emerald-500/30 text-emerald-400' 
                  : 'bg-red-500/10 border border-red-500/30 text-red-400'
              }`}>
                {uploadStatus.success ? (
                  <CheckCircle className="w-5 h-5 flex-shrink-0" />
                ) : (
                  <AlertTriangle className="w-5 h-5 flex-shrink-0" />
                )}
                <span>{uploadStatus.message}</span>
              </div>
            )}
            
            {/* Buttons */}
            <div className="space-y-3">
              <input
                ref={fileInputRef}
                type="file"
                accept=".csv"
                onChange={handleFileChange}
                className="hidden"
              />
              
              <button
                onClick={handleUploadClick}
                disabled={uploadMutation.isPending}
                className="w-full flex items-center justify-center gap-3 px-6 py-4 bg-slate-800 hover:bg-slate-700 border border-slate-700/50 rounded-xl font-semibold text-white transition-all disabled:opacity-50"
              >
                {uploadMutation.isPending ? (
                  <>
                    <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                    </svg>
                    Uploading...
                  </>
                ) : (
                  <>
                    <FileUp className="w-5 h-5" />
                    Upload Catapult CSV
                  </>
                )}
              </button>

              <div className="flex items-center gap-3 text-xs text-slate-500 py-2">
                <div className="flex-1 h-px bg-slate-800" />
                <span>or try demo data</span>
                <div className="flex-1 h-px bg-slate-800" />
              </div>
              
              <button
                onClick={() => loadSampleMutation.mutate()}
                disabled={loadSampleMutation.isPending}
                className="w-full flex items-center justify-center gap-3 px-6 py-4 bg-slate-800/50 hover:bg-slate-800 border border-slate-700/50 rounded-xl font-medium text-slate-300 transition-all disabled:opacity-50"
              >
                {loadSampleMutation.isPending ? (
                  <>
                    <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                    </svg>
                    Loading...
                  </>
                ) : (
                  <>
                    <Database className="w-5 h-5" />
                    Load Demo Data
                    <span className="px-2 py-0.5 bg-slate-700/50 text-slate-400 text-[10px] rounded-full">24 players</span>
                  </>
                )}
              </button>
            </div>

            {/* Features */}
            <div className="mt-8 pt-6 border-t border-slate-800 grid grid-cols-3 gap-4 text-center">
              <div className="p-3">
                <Shield className="w-5 h-5 text-slate-400 mx-auto mb-2" />
                <p className="text-[10px] text-slate-500">Injury Risk</p>
              </div>
              <div className="p-3">
                <TrendingUp className="w-5 h-5 text-slate-400 mx-auto mb-2" />
                <p className="text-[10px] text-slate-500">Performance</p>
              </div>
              <div className="p-3">
                <Sparkles className="w-5 h-5 text-slate-400 mx-auto mb-2" />
                <p className="text-[10px] text-slate-500">AI Insights</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Calculate total from risk distribution
  const totalPlayers = riskDistribution ? (riskDistribution.low + riskDistribution.medium + riskDistribution.high) : 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between animate-fade-in">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-3">
            <div className="p-2 bg-slate-800/60 rounded-xl border border-slate-700/50">
              <Activity className="w-5 h-5 text-slate-300" />
            </div>
            Performance Dashboard
          </h1>
          <p className="text-slate-500 text-sm mt-2 ml-12">
            Real-time insights and injury risk prediction
          </p>
        </div>
        
        <div className="flex items-center gap-3">
          <div className="px-3 py-2 bg-slate-800/50 border border-slate-700/50 rounded-lg text-xs text-slate-400 flex items-center gap-2">
            <Calendar className="w-3.5 h-3.5" />
            {new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv"
            onChange={handleFileChange}
            className="hidden"
          />
          <button
            onClick={handleUploadClick}
            className="flex items-center gap-2 px-4 py-2 bg-slate-800/50 hover:bg-slate-800 border border-slate-700/50 rounded-lg text-xs font-medium text-slate-300 hover:text-white transition-all"
          >
            <Upload className="w-3.5 h-3.5" />
            Upload CSV
          </button>
        </div>
      </div>

      {/* Status message */}
      {uploadStatus && (
        <div className={`p-4 rounded-xl flex items-center justify-between text-sm animate-slide-in-up ${
          uploadStatus.success 
            ? 'bg-emerald-500/10 border border-emerald-500/30' 
            : 'bg-red-500/10 border border-red-500/30'
        }`}>
          <div className="flex items-center gap-3">
            {uploadStatus.success ? (
              <CheckCircle className="w-5 h-5 text-emerald-400" />
            ) : (
              <AlertTriangle className="w-5 h-5 text-red-400" />
            )}
            <span className={uploadStatus.success ? 'text-emerald-400' : 'text-red-400'}>
              {uploadStatus.message}
            </span>
          </div>
          <button 
            onClick={() => setUploadStatus(null)}
            className="text-slate-400 hover:text-white text-lg leading-none px-2"
          >
            ×
          </button>
        </div>
      )}

      {/* Important Notice about 45 days */}
      <div className="p-4 bg-slate-800/30 border border-slate-700/50 rounded-xl flex items-start gap-3 animate-fade-in">
        <Clock className="w-5 h-5 text-slate-400 flex-shrink-0 mt-0.5" />
        <div className="text-sm">
          <p className="text-slate-300 font-medium">Risk Assessment Period</p>
          <p className="text-slate-400 text-xs mt-1">
            Injury risk is calculated using only the <strong>last 45 days</strong> of data from today's date. 
            Players without recent training data are marked as <strong className="text-slate-300">Low Risk</strong>.
          </p>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <KPICard
          title="Total Players"
          value={kpis?.totalPlayers ?? 0}
          change={kpis?.totalPlayersChange ?? 0}
          icon={Users}
          subtitle="active roster"
          delay={0}
        />
        <KPICard
          title="Avg Team Load"
          value={`${kpis?.avgTeamLoad ?? 0}`}
          change={kpis?.avgTeamLoadChange ?? 0}
          icon={Zap}
          subtitle="units per session"
          delay={50}
        />
        <KPICard
          title="High Risk Players"
          value={kpis?.highRiskPlayers ?? 0}
          change={0}
          icon={AlertTriangle}
          subtitle="last 45 days"
          variant="warning"
          delay={100}
        />
        <KPICard
          title="Avg Team Speed"
          value={`${kpis?.avgTeamSpeed ?? 0}`}
          change={kpis?.avgTeamSpeedChange ?? 0}
          icon={Gauge}
          subtitle="mph"
          variant="success"
          delay={150}
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Load Chart - Takes more space */}
        <div className="lg:col-span-2">
          <LoadChart data={loadHistory ?? []} />
        </div>
        
        {/* Risk Distribution */}
        <div className="card p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold text-white flex items-center gap-2">
              <Shield className="w-4 h-4 text-slate-400" />
              Risk Distribution
            </h3>
            <span className="text-xs text-slate-500">{totalPlayers} players</span>
          </div>
          
          <RiskDonut data={riskDistribution ?? { low: 0, medium: 0, high: 0 }} />
          
          {/* Legend */}
          <div className="mt-4 grid grid-cols-3 gap-2">
            <div className="text-center p-3 bg-emerald-500/10 rounded-lg">
              <p className="text-2xl font-bold text-emerald-400">{riskDistribution?.low ?? 0}</p>
              <p className="text-[10px] text-slate-500 mt-1">Low Risk</p>
            </div>
            <div className="text-center p-3 bg-yellow-500/10 rounded-lg">
              <p className="text-2xl font-bold text-yellow-400">{riskDistribution?.medium ?? 0}</p>
              <p className="text-[10px] text-slate-500 mt-1">Medium</p>
            </div>
            <div className="text-center p-3 bg-red-500/10 rounded-lg">
              <p className="text-2xl font-bold text-red-400">{riskDistribution?.high ?? 0}</p>
              <p className="text-[10px] text-slate-500 mt-1">High Risk</p>
            </div>
          </div>
        </div>
      </div>

      {/* Player Lists */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <PlayerList
          title="High Risk Players"
          players={highRiskPlayers ?? []}
          type="risk"
          viewAllLink="/players?filter=high-risk"
        />
        <PlayerList
          title="Top Performers"
          players={topPerformers ?? []}
          type="top"
          viewAllLink="/players"
        />
      </div>
    </div>
  );
}
