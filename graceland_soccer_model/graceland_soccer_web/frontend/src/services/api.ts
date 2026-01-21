import axios from 'axios';
import { useQuery } from '@tanstack/react-query';
import type {
  DashboardKPIs,
  RiskDistribution,
  LoadHistory,
  Player,
  PlayerDetail,
  LoadPrediction,
  RiskPrediction,
  TrainingResult,
  UploadResult,
  ApiResponse,
} from '../types';

const api = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Dashboard endpoints
export const dashboardApi = {
  getKPIs: async (): Promise<DashboardKPIs> => {
    const { data } = await api.get<ApiResponse<DashboardKPIs>>('/dashboard/kpis');
    return data.data!;
  },

  getLoadHistory: async (days: number = 15): Promise<LoadHistory[]> => {
    const { data } = await api.get<ApiResponse<LoadHistory[]>>(`/dashboard/load-history?days=${days}`);
    return data.data!;
  },

  getRiskDistribution: async (): Promise<RiskDistribution> => {
    const { data } = await api.get<ApiResponse<RiskDistribution>>('/dashboard/risk-distribution');
    return data.data!;
  },
};

// Players endpoints
export const playersApi = {
  getAll: async (): Promise<Player[]> => {
    const { data } = await api.get<ApiResponse<Player[]>>('/players');
    return data.data!;
  },

  getById: async (id: string): Promise<PlayerDetail> => {
    const { data } = await api.get<ApiResponse<PlayerDetail>>(`/players/${id}`);
    return data.data!;
  },

  getHighRisk: async (): Promise<Player[]> => {
    const { data } = await api.get<ApiResponse<Player[]>>('/players/high-risk');
    return data.data!;
  },

  getTopPerformers: async (limit: number = 5): Promise<Player[]> => {
    const { data } = await api.get<ApiResponse<Player[]>>(`/players/top-performers?limit=${limit}`);
    return data.data!;
  },

  getDetail: async (playerId: string): Promise<any> => {
    const { data } = await api.get<ApiResponse<any>>(`/players/${playerId}`);
    return data.data!;
  },

  getExcluded: async (): Promise<string[]> => {
    const { data } = await api.get<ApiResponse<string[]>>('/players/excluded');
    return data.data!;
  },

  excludePlayer: async (playerName: string): Promise<{ message: string; playerName: string }> => {
    const { data } = await api.post<ApiResponse<{ message: string; playerName: string }>>('/players/exclude', { playerName });
    return data.data!;
  },

  restorePlayer: async (playerName: string): Promise<{ message: string; playerName: string }> => {
    const { data } = await api.post<ApiResponse<{ message: string; playerName: string }>>('/players/restore', { playerName });
    return data.data!;
  },

  deletePlayer: async (playerId: string): Promise<{ message: string; playerId: string }> => {
    const { data } = await api.delete<ApiResponse<{ message: string; playerId: string }>>(`/players/${playerId}`);
    return data.data!;
  },
};

// Analysis endpoints
export const analysisApi = {
  predictLoad: async (playerId: string, features: Record<string, number>): Promise<LoadPrediction> => {
    const { data } = await api.post<ApiResponse<LoadPrediction>>('/analysis/predict-load', {
      playerId,
      features,
    });
    return data.data!;
  },

  predictRisk: async (playerId: string): Promise<RiskPrediction> => {
    const { data } = await api.post<ApiResponse<RiskPrediction>>('/analysis/predict-risk', {
      playerId,
    });
    return data.data!;
  },

  comparePlayersLoad: async (playerIds: string[]): Promise<LoadPrediction[]> => {
    const { data } = await api.post<ApiResponse<LoadPrediction[]>>('/analysis/compare', {
      playerIds,
    });
    return data.data!;
  },

  getOllamaStatus: async (): Promise<any> => {
    const { data } = await api.get<ApiResponse<any>>('/analysis/ollama-status');
    return data.data!;
  },

  getAIRecommendations: async (playerId: string): Promise<any> => {
    const { data } = await api.post<ApiResponse<any>>('/analysis/ai-recommendations', { playerId });
    return data.data!;
  },
};

// Training endpoints
export const trainingApi = {
  trainLoadModel: async (algorithm: string): Promise<TrainingResult> => {
    const { data } = await api.post<ApiResponse<TrainingResult>>('/training/train-load', {
      algorithm,
    });
    return data.data!;
  },

  trainRiskModel: async (algorithm: string): Promise<TrainingResult> => {
    const { data } = await api.post<ApiResponse<TrainingResult>>('/training/train-risk', {
      algorithm,
    });
    return data.data!;
  },

  getModelStatus: async (): Promise<any> => {
    const { data } = await api.get<ApiResponse<any>>('/training/status');
    return data.data!;
  },

  predictLoad: async (params: { playerId: string; sessionType: string; features: Record<string, number> }): Promise<any> => {
    const { data } = await api.post<ApiResponse<any>>('/training/predict-load', params);
    return data.data!;
  },
};

// Data endpoints
export const dataApi = {
  upload: async (file: File): Promise<UploadResult> => {
    const formData = new FormData();
    formData.append('file', file);
    const { data } = await api.post<ApiResponse<UploadResult>>('/data/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return data.data!;
  },

  getStatus: async (): Promise<{ loaded: boolean; rowCount: number; players: string[] }> => {
    const { data } = await api.get<ApiResponse<{ loaded: boolean; rowCount: number; players: string[] }>>('/data/status');
    return data.data!;
  },

  loadSample: async (): Promise<UploadResult> => {
    const { data } = await api.post<ApiResponse<UploadResult>>('/data/load-sample');
    return data.data!;
  },

  getAudit: async (): Promise<any> => {
    const { data } = await api.get<ApiResponse<any>>('/data/audit');
    return data.data!;
  },

  cleanOutliers: async (method: string = 'iqr', threshold: number = 1.5): Promise<any> => {
    const { data } = await api.post<ApiResponse<any>>('/data/clean-outliers', { method, threshold });
    return data.data!;
  },

  resetData: async (): Promise<any> => {
    const { data } = await api.post<ApiResponse<any>>('/data/reset');
    return data.data!;
  },
};

// Settings endpoints
export const settingsApi = {
  getDateReference: async (): Promise<{ useTodayAsReference: boolean; description: string }> => {
    const { data } = await api.get<ApiResponse<{ useTodayAsReference: boolean; description: string }>>('/settings/date-reference');
    return data.data!;
  },

  setDateReference: async (useTodayAsReference: boolean): Promise<any> => {
    const { data } = await api.post<ApiResponse<any>>('/settings/date-reference', { useTodayAsReference });
    return data.data!;
  },
};

// React Query hooks for data status
export const useDataStatus = () => {
  return useQuery({
    queryKey: ['data', 'status'],
    queryFn: dataApi.getStatus,
  });
};

export default api;
