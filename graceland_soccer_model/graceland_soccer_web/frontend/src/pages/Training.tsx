import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { 
  Dumbbell, 
  CheckCircle, 
  XCircle, 
  Loader2, 
  TrendingUp, 
  Target,
  Brain,
  Cpu,
  BarChart3,
  AlertTriangle,
  Info,
  ChevronRight,
  Sparkles,
  RefreshCw
} from 'lucide-react';
import { trainingApi, useDataStatus } from '../services/api';

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

export default function Training() {
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

  // No data loaded state
  if (!dataStatus?.loaded) {
    return (
      <div className="min-h-[70vh] flex items-center justify-center animate-fade-in">
        <div className="card p-8 max-w-md text-center">
          <div className="w-16 h-16 mx-auto mb-6 bg-slate-800/60 border border-slate-700/50 rounded-2xl flex items-center justify-center">
            <AlertTriangle className="w-8 h-8 text-slate-300" />
          </div>
          <h2 className="text-xl font-bold text-white mb-2">No Data Loaded</h2>
          <p className="text-slate-400 text-sm mb-6">
            Load CSV data first before training models. Go to Dashboard to upload your Catapult data.
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
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-3">
            <Brain className="w-7 h-7 text-slate-400" />
            Model Training
          </h1>
          <p className="text-slate-500 text-sm mt-1">
            Retrain ML models with current data
          </p>
        </div>
        <div className="flex items-center gap-2 px-3 py-1.5 bg-slate-800/50 rounded-lg border border-slate-700/50">
          <Cpu className="w-4 h-4 text-slate-400" />
          <span className="text-xs text-slate-400">
            {dataStatus?.rowCount?.toLocaleString()} samples available
          </span>
        </div>
      </div>

      {/* Model Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Load Model Status */}
        <div className="card p-5">
          <div className="flex items-center gap-4">
            <div className={`
              w-14 h-14 rounded-xl flex items-center justify-center
              ${modelStatus?.loadModel 
                ? 'bg-slate-800/60 border border-slate-600' 
                : 'bg-slate-800 border border-slate-700'
              }
            `}>
              {modelStatus?.loadModel ? (
                <CheckCircle className="w-7 h-7 text-white" />
              ) : (
                <XCircle className="w-7 h-7 text-slate-500" />
              )}
            </div>
            <div className="flex-1">
              <p className="font-semibold text-white">Player Load Model</p>
              <p className="text-sm text-slate-500">
                {modelStatus?.loadModel ? (
                  <span className="text-emerald-400">
                    {modelStatus.loadModelDetails?.algorithm || 'GradientBoostingRegressor'}
                  </span>
                ) : (
                  'Not trained yet'
                )}
              </p>
              {modelStatus?.loadModelDetails?.metrics && (
                <p className="text-xs text-slate-500 mt-1">
                  R² = {modelStatus.loadModelDetails.metrics.r2Score || modelStatus.loadModelDetails.metrics.R2 || 'N/A'}
                </p>
              )}
            </div>
          </div>
        </div>

        {/* Risk Model Status */}
        <div className="card p-5">
          <div className="flex items-center gap-4">
            <div className={`
              w-14 h-14 rounded-xl flex items-center justify-center
              ${modelStatus?.riskModel 
                ? 'bg-slate-800/60 border border-slate-600' 
                : 'bg-slate-800 border border-slate-700'
              }
            `}>
              {modelStatus?.riskModel ? (
                <CheckCircle className="w-7 h-7 text-white" />
              ) : (
                <XCircle className="w-7 h-7 text-slate-500" />
              )}
            </div>
            <div className="flex-1">
              <p className="font-semibold text-white">Injury Risk Model</p>
              <p className="text-sm text-slate-500">
                {modelStatus?.riskModel ? (
                  <span className="text-emerald-400">
                    {modelStatus.riskModelDetails?.algorithm || 'LGBMClassifier'}
                  </span>
                ) : (
                  'Not trained yet'
                )}
              </p>
              {modelStatus?.riskModelDetails?.metrics && (
                <p className="text-xs text-slate-500 mt-1">
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
        <div className="card p-6">
          <div className="flex items-center gap-3 mb-5">
            <div className="p-3 rounded-xl bg-slate-800/60 border border-slate-700/50">
              <TrendingUp className="w-6 h-6 text-slate-300" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-white">Player Load Prediction</h2>
              <p className="text-sm text-slate-500">GradientBoostingRegressor</p>
            </div>
          </div>

          {/* Info Box */}
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
            className="w-full py-3.5 btn-primary rounded-xl font-semibold text-white flex items-center justify-center gap-2 disabled:opacity-50"
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

          {/* Training Results */}
          {trainLoadMutation.data && (
            <div className="mt-4 p-4 bg-emerald-500/10 border border-emerald-500/20 rounded-xl animate-slide-in-up">
              <div className="flex items-center gap-2 mb-3">
                <CheckCircle className="w-4 h-4 text-emerald-400" />
                <p className="text-sm font-semibold text-emerald-400">Training Complete!</p>
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
        <div className="card p-6">
          <div className="flex items-center gap-3 mb-5">
            <div className="p-3 rounded-xl bg-slate-800/60 border border-slate-700/50">
              <Target className="w-6 h-6 text-slate-300" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-white">Injury Risk Classification</h2>
              <p className="text-sm text-slate-500">LGBMClassifier (LightGBM)</p>
            </div>
          </div>

          {/* Info Box */}
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

          {/* Training Results */}
          {trainRiskMutation.data && (
            <div className="mt-4 p-4 bg-emerald-500/10 border border-emerald-500/20 rounded-xl animate-slide-in-up">
              <div className="flex items-center gap-2 mb-3">
                <CheckCircle className="w-4 h-4 text-emerald-400" />
                <p className="text-sm font-semibold text-emerald-400">Training Complete!</p>
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

      {/* Model Information */}
      <div className="card p-6">
        <div className="flex items-center gap-3 mb-4">
          <BarChart3 className="w-5 h-5 text-slate-400" />
          <h3 className="font-semibold text-white">Model Information</h3>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div className="p-4 bg-slate-800/30 rounded-xl">
            <p className="text-slate-300 font-medium mb-2">Player Load Model</p>
            <ul className="space-y-1 text-slate-400">
              <li>• Algorithm: GradientBoostingRegressor</li>
              <li>• Preprocessing: ColumnTransformer + SelectFromModel</li>
              <li>• Standard preprocessing pipeline</li>
            </ul>
          </div>
          <div className="p-4 bg-slate-800/30 rounded-xl">
            <p className="text-orange-400 font-medium mb-2">Injury Risk Model</p>
            <ul className="space-y-1 text-slate-400">
              <li>• Algorithm: LGBMClassifier (LightGBM)</li>
              <li>• Preprocessing: ColumnTransformer + SelectKBest</li>
              <li>• Standard preprocessing pipeline</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
