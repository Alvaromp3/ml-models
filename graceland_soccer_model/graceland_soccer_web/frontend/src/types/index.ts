// Player types
export interface Player {
  id: string;
  name: string;
  position: string;
  number: number;
  riskLevel: 'low' | 'medium' | 'high';
  avgLoad: number;
  avgSpeed: number;
  sessions: number;
  lastSession?: string;
}

export interface PlayerDetail extends Player {
  metrics: PlayerMetrics;
  history: SessionData[];
}

export interface PlayerMetrics {
  playerLoad: number;
  distance: number;
  sprintDistance: number;
  topSpeed: number;
  maxAcceleration: number;
  maxDeceleration: number;
  workRatio: number;
  energy: number;
  hrLoad: number;
  impacts: number;
  powerPlays: number;
}

// Session and load data
export interface SessionData {
  date: string;
  sessionTitle: string;
  playerLoad: number;
  distance: number;
  duration: number;
  avgSpeed: number;
  topSpeed: number;
}

export interface LoadHistory {
  date: string;
  avgLoad: number;
  sessionCount: number;
}

// Dashboard KPIs
export interface DashboardKPIs {
  totalPlayers: number;
  totalPlayersChange: number;
  avgTeamLoad: number;
  avgTeamLoadChange: number;
  highRiskPlayers: number;
  highRiskPlayersChange: number;
  avgTeamSpeed: number;
  avgTeamSpeedChange: number;
}

export interface RiskDistribution {
  low: number;
  medium: number;
  high: number;
}

// Analysis and predictions
export interface LoadPrediction {
  playerId: string;
  playerName: string;
  predictedLoad: number;
  confidence: number;
  features: Record<string, number>;
}

export interface RiskPrediction {
  playerId: string;
  playerName: string;
  riskLevel: 'low' | 'medium' | 'high';
  probability: number;
  factors: string[];
  recommendations: string[];
  hasRecentData?: boolean;
  recentSessionCount?: number;
}

// Training
export interface ModelMetrics {
  r2Score: number;
  mae: number;
  rmse: number;
  mse: number;
}

export interface ClassificationMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
}

export interface TrainingResult {
  modelType: 'regression' | 'classification';
  algorithm: string;
  metrics: Record<string, number>;
  trainingTime: number;
  timestamp: string;
}

// API responses
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

// Data upload
export interface UploadResult {
  rowCount: number;
  columnCount: number;
  columns: string[];
  players: string[];
  dateRange: {
    start: string;
    end: string;
  };
}
