import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { 
  CheckCircle, 
  XCircle, 
  Loader2, 
  TrendingUp, 
  Target,
  Cpu,
  AlertTriangle,
  Info,
  ChevronRight,
  RefreshCw
} from 'lucide-react';
import { trainingApi, useDataStatus } from '../../services/api';

interface ModelDetails {
  trained: boolean;
  algorithm: string | null;
  metrics: Record<string, number> | null;
  features: number;
}

interface ModelStatusResponse {
  loadModel: boolean;
  riskModel: boolean;
  loadModelDetails: ModelDetails;
  riskModelDetails: ModelDetails;
}

export default function TrainingContent() {
  const queryClient = useQueryClient();
  const { data: dataStatus } = useDataStatus();
  
  const { data: modelStatus, isLoading: statusLoading } = useQuery<ModelStatusResponse>({
    queryKey: ['training', 'status'],
    queryFn: trainingApi.getModelStatus,
  });

  const trainLoadMutation = useMutation({
    mutationFn: () => trainingApi.trainLoadModel('gradient_boosting'),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['training', 'status'] }),
  });

  const trainRiskMutation = useMutation({
    mutationFn: () => trainingApi.trainRiskModel('lightgbm'),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['training', 'status'] }),
  });

  if (!dataStatus?.loaded) {
    return (
      <div className="panel panel--elevated p-8 text-center">
        <div className="w-16 h-16 mx-auto mb-6 bg-slate-800/60 border border-slate-700/50 rounded-2xl flex items-center justify-center">
          <AlertTriangle className="w-8 h-8 text-slate-300" />
        </div>
        <h2 className="text-xl font-bold text-white mb-2">No Data Loaded</h2>
        <p className="text-slate-400 text-sm mb-6">
          Upload a CSV in the Dashboard to train the models.
        </p>
        <a 
          href="/"
          className="inline-flex items-center gap-2 px-5 py-2.5 btn-primary rounded-xl font-medium text-white text-sm"
        >
          Go to Dashboard
          <ChevronRight className="w-4 h-4" />
        </a>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Model Status Cards - Estilo fichas escritas */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Load Model Status */}
        <div className="panel p-5" style={{ transform: 'rotate(-0.3deg)' }}>
          <div className="flex items-center gap-4">
            <div className={`
              w-16 h-16 flex items-center justify-center
              ${modelStatus?.loadModel 
                ? 'bg-[#e8f5e3] border-3 border-[#4a7c2a]' 
                : 'bg-[#f5f0e3] border-3 border-[#d4c5b0]'
              }
            `} style={{
              borderRadius: '8px',
              boxShadow: '2px 2px 6px rgba(44, 36, 22, 0.15)',
              transform: 'rotate(2deg)',
              borderWidth: '3px'
            }}>
              {modelStatus?.loadModel ? (
                <CheckCircle className="w-8 h-8" style={{ color: '#2d5016' }} strokeWidth={2.5} />
              ) : (
                <XCircle className="w-8 h-8" style={{ color: '#8b7355' }} strokeWidth={2} />
              )}
            </div>
            <div className="flex-1">
              <p className="font-bold text-lg text-[#2c2416] handwritten" style={{ 
                letterSpacing: '0.5px',
                transform: 'rotate(0.2deg)'
              }}>
                Player Load Model
              </p>
              <p className="text-sm text-[#5a4a3a] mt-1 italic" style={{ transform: 'rotate(-0.2deg)' }}>
                {modelStatus?.loadModel ? (
                  <span style={{ color: '#2d5016', fontWeight: 600 }}>
                    {modelStatus.loadModelDetails?.algorithm || 'GradientBoostingRegressor'}
                  </span>
                ) : (
                  <span style={{ color: '#8b7355' }}>Not trained yet</span>
                )}
              </p>
              {modelStatus?.loadModelDetails?.metrics && (
                <p className="text-xs text-[#6b5d47] mt-2 font-semibold" style={{ 
                  transform: 'rotate(0.3deg)',
                  borderBottom: '2px solid #d4c5b0',
                  paddingBottom: '4px',
                  display: 'inline-block'
                }}>
                  R² = {modelStatus.loadModelDetails.metrics.r2Score || modelStatus.loadModelDetails.metrics.R2 || 'N/A'}
                </p>
              )}
            </div>
          </div>
        </div>

        {/* Risk Model Status */}
        <div className="panel p-5" style={{ transform: 'rotate(0.3deg)' }}>
          <div className="flex items-center gap-4">
            <div className={`
              w-16 h-16 flex items-center justify-center
              ${modelStatus?.riskModel 
                ? 'bg-[#f5e8e3] border-3 border-[#b8651a]' 
                : 'bg-[#f5f0e3] border-3 border-[#d4c5b0]'
              }
            `} style={{
              borderRadius: '8px',
              boxShadow: '2px 2px 6px rgba(44, 36, 22, 0.15)',
              transform: 'rotate(-2deg)',
              borderWidth: '3px'
            }}>
              {modelStatus?.riskModel ? (
                <CheckCircle className="w-8 h-8" style={{ color: '#8b4513' }} strokeWidth={2.5} />
              ) : (
                <XCircle className="w-8 h-8" style={{ color: '#8b7355' }} strokeWidth={2} />
              )}
            </div>
            <div className="flex-1">
              <p className="font-bold text-lg text-[#2c2416] handwritten" style={{ 
                letterSpacing: '0.5px',
                transform: 'rotate(-0.2deg)'
              }}>
                Injury Risk Model
              </p>
              <p className="text-sm text-[#5a4a3a] mt-1 italic" style={{ transform: 'rotate(0.2deg)' }}>
                {modelStatus?.riskModel ? (
                  <span style={{ color: '#8b4513', fontWeight: 600 }}>
                    {modelStatus.riskModelDetails?.algorithm || 'LGBMClassifier'}
                  </span>
                ) : (
                  <span style={{ color: '#8b7355' }}>Not trained yet</span>
                )}
              </p>
              {modelStatus?.riskModelDetails?.metrics && (
                <p className="text-xs text-[#6b5d47] mt-2 font-semibold" style={{ 
                  transform: 'rotate(-0.3deg)',
                  borderBottom: '2px solid #d4c5b0',
                  paddingBottom: '4px',
                  display: 'inline-block'
                }}>
                  Accuracy = {modelStatus.riskModelDetails.metrics.accuracy || modelStatus.riskModelDetails.metrics.Accuracy || 'N/A'}
                </p>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Training Panels */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Load Model Training */}
        <div className="panel panel--elevated p-6">
          <div className="flex items-center gap-3 mb-5">
            <div className="p-3 rounded-xl bg-slate-800/60 border border-slate-700/50">
              <TrendingUp className="w-6 h-6 text-slate-300" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-white">Player Load Prediction</h2>
              <p className="text-sm text-slate-500">GradientBoostingRegressor</p>
            </div>
          </div>

          <div className="mb-5 p-3 bg-slate-800/30 border border-slate-700/50 rounded-xl">
            <div className="flex items-start gap-2">
              <Info className="w-4 h-4 text-slate-400 flex-shrink-0 mt-0.5" />
              <div className="text-xs text-slate-400">
                <p>Predicts Player Load based on metrics like duration, distance, speed, and accelerations.</p>
              </div>
            </div>
          </div>

          <button
            onClick={() => trainLoadMutation.mutate()}
            disabled={trainLoadMutation.isPending}
            className="btn btn--primary w-full py-3.5 gap-2 disabled:opacity-50"
          >
            {trainLoadMutation.isPending ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Training...
              </>
            ) : (
              <>
                <RefreshCw className="w-5 h-5" />
                Retrain Model
              </>
            )}
          </button>

          {trainLoadMutation.data && (
            <div className="mt-4 p-4 bg-[#1e40af]/10 border border-[#1e40af]/20 rounded-xl animate-slide-in-up">
              <div className="flex items-center gap-2 mb-3">
                <CheckCircle className="w-4 h-4 text-[#1e40af]" />
                <p className="text-sm font-semibold text-[#1e40af]">Training Complete!</p>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div className="p-3 bg-slate-800/50 rounded-lg">
                  <p className="text-[10px] text-slate-500 uppercase tracking-wider">R² Score</p>
                  <p className="text-lg font-bold text-white">{trainLoadMutation.data.metrics.r2Score}</p>
                </div>
                <div className="p-3 bg-slate-800/50 rounded-lg">
                  <p className="text-[10px] text-slate-500 uppercase tracking-wider">MAE</p>
                  <p className="text-lg font-bold text-white">{trainLoadMutation.data.metrics.mae}</p>
                </div>
                <div className="p-3 bg-slate-800/50 rounded-lg">
                  <p className="text-[10px] text-slate-500 uppercase tracking-wider">RMSE</p>
                  <p className="text-lg font-bold text-white">{trainLoadMutation.data.metrics.rmse}</p>
                </div>
                <div className="p-3 bg-slate-800/50 rounded-lg">
                  <p className="text-[10px] text-slate-500 uppercase tracking-wider">Time</p>
                  <p className="text-lg font-bold text-white">{trainLoadMutation.data.trainingTime}s</p>
                </div>
              </div>
            </div>
          )}

          {trainLoadMutation.isError && (
            <div className="mt-4 p-4 bg-red-500/10 border border-red-500/20 rounded-xl">
              <div className="flex items-center gap-2">
                <XCircle className="w-4 h-4 text-red-400" />
                <p className="text-sm text-red-400">
                  Training failed: {(trainLoadMutation.error as Error)?.message || 'Unknown error'}
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Risk Model Training */}
        <div className="panel panel--elevated p-6">
          <div className="flex items-center gap-3 mb-5">
            <div className="p-3 rounded-xl bg-slate-800/60 border border-slate-700/50">
              <Target className="w-6 h-6 text-slate-300" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-white">Injury Risk Classification</h2>
              <p className="text-sm text-slate-500">LGBMClassifier (LightGBM)</p>
            </div>
          </div>

          <div className="mb-5 p-3 bg-orange-500/10 border border-orange-500/20 rounded-xl">
            <div className="flex items-start gap-2">
              <Info className="w-4 h-4 text-orange-400 flex-shrink-0 mt-0.5" />
              <div className="text-xs text-slate-400">
                <p>Classifies players into Low, Medium, or High injury risk using LightGBM.</p>
              </div>
            </div>
          </div>

          <button
            onClick={() => trainRiskMutation.mutate()}
            disabled={trainRiskMutation.isPending}
            className="w-full py-3.5 bg-slate-800 hover:bg-slate-700 border border-slate-700/50 rounded-xl font-semibold text-white flex items-center justify-center gap-2 disabled:opacity-50 transition-all"
          >
            {trainRiskMutation.isPending ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Training...
              </>
            ) : (
              <>
                <RefreshCw className="w-5 h-5" />
                Retrain Model
              </>
            )}
          </button>

          {trainRiskMutation.data && (
            <div className="mt-4 p-4 bg-[#1e40af]/10 border border-[#1e40af]/20 rounded-xl animate-slide-in-up">
              <div className="flex items-center gap-2 mb-3">
                <CheckCircle className="w-4 h-4 text-[#1e40af]" />
                <p className="text-sm font-semibold text-[#1e40af]">Training Complete!</p>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div className="p-3 bg-slate-800/50 rounded-lg">
                  <p className="text-[10px] text-slate-500 uppercase tracking-wider">Accuracy</p>
                  <p className="text-lg font-bold text-white">{(trainRiskMutation.data.metrics.accuracy * 100).toFixed(1)}%</p>
                </div>
                <div className="p-3 bg-slate-800/50 rounded-lg">
                  <p className="text-[10px] text-slate-500 uppercase tracking-wider">Precision</p>
                  <p className="text-lg font-bold text-white">{(trainRiskMutation.data.metrics.precision * 100).toFixed(1)}%</p>
                </div>
                <div className="p-3 bg-slate-800/50 rounded-lg">
                  <p className="text-[10px] text-slate-500 uppercase tracking-wider">Recall</p>
                  <p className="text-lg font-bold text-white">{(trainRiskMutation.data.metrics.recall * 100).toFixed(1)}%</p>
                </div>
                <div className="p-3 bg-slate-800/50 rounded-lg">
                  <p className="text-[10px] text-slate-500 uppercase tracking-wider">F1 Score</p>
                  <p className="text-lg font-bold text-white">{(trainRiskMutation.data.metrics.f1Score * 100).toFixed(1)}%</p>
                </div>
              </div>
            </div>
          )}

          {trainRiskMutation.isError && (
            <div className="mt-4 p-4 bg-red-500/10 border border-red-500/20 rounded-xl">
              <div className="flex items-center gap-2">
                <XCircle className="w-4 h-4 text-red-400" />
                <p className="text-sm text-red-400">
                  Training failed: {(trainRiskMutation.error as Error)?.message || 'Unknown error'}
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Info */}
      <div className="panel p-4 border-[var(--border-default)] bg-[var(--bg-subtle)]">
        <div className="flex items-start gap-3">
          <Info className="w-5 h-5 text-slate-400 flex-shrink-0 mt-0.5" />
          <div className="text-sm text-slate-400">
            <p className="text-slate-300 font-medium mb-1">About Model Training</p>
            <p>
              Retrain models when you have new data to improve predictions. Training uses all available data and may take a few moments.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
