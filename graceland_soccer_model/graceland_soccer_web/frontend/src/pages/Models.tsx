import { useState, useMemo } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { CheckCircle, XCircle, Loader2, RefreshCw, ChevronRight } from 'lucide-react';
import { getModelStatus, trainingApi, useDataStatus } from '../services/api';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  LineChart,
  Line,
  CartesianGrid,
  Legend
} from 'recharts';

type ModelTab = 'explanation' | 'training';

export default function Models() {
  const queryClient = useQueryClient();
  const [activeTab, setActiveTab] = useState<ModelTab>('explanation');
  const { data: dataStatus } = useDataStatus();

  const { data: modelStatus, isLoading } = useQuery({
    queryKey: ['modelStatus'],
    queryFn: getModelStatus,
  });

  const { data: trainingStatus } = useQuery({
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

  const loadFeatures = [
    {
      name: 'Duration',
      description: 'Total duration of the training session',
      importance: 'High',
      reason: 'Longer sessions generate greater accumulated load on the player',
      weight: 85
    },
    {
      name: 'Distance (miles)',
      description: 'Total distance covered during the session',
      importance: 'High',
      reason: 'Greater distance implies greater physical effort and metabolic load',
      weight: 88
    },
    {
      name: 'Sprint Distance (yards)',
      description: 'Distance covered in high-intensity sprints',
      importance: 'Very High',
      reason: 'Sprints generate significant neuromuscular load and fatigue',
      weight: 95
    },
    {
      name: 'Top Speed (mph)',
      description: 'Maximum speed reached',
      importance: 'High',
      reason: 'Maximum speeds indicate maximum efforts that increase load',
      weight: 82
    },
    {
      name: 'Max Acceleration (yd/s/s)',
      description: 'Maximum acceleration during the session',
      importance: 'High',
      reason: 'Maximum accelerations generate high muscular and metabolic load',
      weight: 80
    },
    {
      name: 'Max Deceleration (yd/s/s)',
      description: 'Maximum deceleration during the session',
      importance: 'Medium',
      reason: 'Decelerations generate eccentric load on muscles',
      weight: 65
    },
    {
      name: 'Work Ratio',
      description: 'Work ratio (relationship between work and recovery)',
      importance: 'Very High',
      reason: 'Indicates work intensity and fatigue accumulation',
      weight: 92
    },
    {
      name: 'Energy (kcal)',
      description: 'Total energy consumed during the session',
      importance: 'High',
      reason: 'Reflects total energy expenditure and metabolic load',
      weight: 78
    },
    {
      name: 'Hr Load',
      description: 'Load based on heart rate',
      importance: 'Medium',
      reason: 'Indicates cardiovascular response to effort',
      weight: 70
    },
    {
      name: 'Impacts',
      description: 'Number of impacts received',
      importance: 'Medium',
      reason: 'Repeated impacts can contribute to accumulated load',
      weight: 68
    },
    {
      name: 'Power Plays',
      description: 'High-power plays',
      importance: 'Medium',
      reason: 'Indicates moments of maximum intensity during the session',
      weight: 72
    },
    {
      name: 'Power Score (w/kg)',
      description: 'Power score relative to weight',
      importance: 'High',
      reason: 'Reflects power generated and neuromuscular load',
      weight: 83
    },
    {
      name: 'Distance Per Min (yd/min)',
      description: 'Average distance per minute',
      importance: 'Medium',
      reason: 'Indicates average work pace during the session',
      weight: 60
    }
  ];

  const riskFeatures = [
    {
      name: 'Player Load',
      description: 'Total player load (main objective)',
      importance: 'Critical',
      reason: 'It is the main metric that determines the injury risk level',
      weight: 100
    },
    {
      name: 'Work Ratio',
      description: 'Work ratio vs recovery',
      importance: 'Very High',
      reason: 'High ratios indicate accumulated fatigue and greater overload risk',
      weight: 90
    },
    {
      name: 'Sprint Distance',
      description: 'Distance in high-intensity sprints',
      importance: 'High',
      reason: 'Excessive sprints increase the risk of muscle injuries',
      weight: 85
    },
    {
      name: 'Top Speed',
      description: 'Maximum speed reached',
      importance: 'High',
      reason: 'Repeated maximum speeds increase injury risk',
      weight: 80
    },
    {
      name: 'Distance',
      description: 'Total distance covered',
      importance: 'Medium',
      reason: 'Very high distances may indicate cumulative overload',
      weight: 70
    }
  ];

  const tabs: { id: ModelTab; label: string }[] = [
    { id: 'explanation', label: 'Model Overview' },
    { id: 'training', label: 'Training' },
  ];

  // Chart data for feature importance
  const loadFeatureChartData = useMemo(() => {
    return loadFeatures
      .sort((a, b) => b.weight - a.weight)
      .slice(0, 8)
      .map(f => ({
        name: f.name.split('(')[0].trim(),
        importance: f.weight,
        category: f.importance === 'Very High' ? 'Critical' : f.importance
      }));
  }, []);

  const riskFeatureChartData = useMemo(() => {
    return riskFeatures.map(f => ({
      name: f.name,
      importance: f.weight,
      category: f.importance
    }));
  }, []);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="panel panel--elevated p-6">
        <h1 className="text-2xl font-bold text-[var(--text-primary)]">
          Machine Learning Models
        </h1>
        <p className="text-sm text-[var(--text-secondary)] mt-1">
          Performance prediction and risk classification
        </p>
        <div className="flex gap-2 mt-4 pt-4 border-t border-[var(--border-subtle)]">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`
                px-4 py-2 text-sm font-medium rounded-lg transition-colors
                ${activeTab === tab.id
                  ? 'bg-[var(--accent-performance)] text-white'
                  : 'bg-[var(--bg-subtle)] text-[var(--text-secondary)] hover:bg-[var(--bg-elevated)] hover:text-[var(--text-primary)]'
                }
              `}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Tab Content */}
      {activeTab === 'explanation' && (
        <div className="space-y-6">
          {/* Model status summary */}
          {!isLoading && modelStatus && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="card p-5 bg-white">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-semibold text-[var(--text-primary)]">Regression Model</h3>
                  {modelStatus.loadModel && (
                    <span className="text-xs font-medium px-2 py-1 rounded bg-[var(--accent-performance-muted)] text-[var(--accent-performance)]">
                      Trained
                    </span>
                  )}
                </div>
                {modelStatus.loadModelDetails?.metrics && (
                  <div className="grid grid-cols-3 gap-2 text-center">
                    <div className="p-2 rounded-lg bg-[var(--bg-subtle)]">
                      <p className="text-[10px] uppercase tracking-wider text-[var(--text-tertiary)]">R²</p>
                      <p className="text-lg font-bold text-[var(--text-primary)]">{modelStatus.loadModelDetails.metrics.r2Score?.toFixed(3) || 'N/A'}</p>
                    </div>
                    <div className="p-2 rounded-lg bg-[var(--bg-subtle)]">
                      <p className="text-[10px] uppercase tracking-wider text-[var(--text-tertiary)]">MAE</p>
                      <p className="text-lg font-bold text-[var(--text-primary)]">{modelStatus.loadModelDetails.metrics.mae?.toFixed(2) || 'N/A'}</p>
                    </div>
                    <div className="p-2 rounded-lg bg-[var(--bg-subtle)]">
                      <p className="text-[10px] uppercase tracking-wider text-[var(--text-tertiary)]">RMSE</p>
                      <p className="text-lg font-bold text-[var(--text-primary)]">{modelStatus.loadModelDetails.metrics.rmse?.toFixed(2) || 'N/A'}</p>
                    </div>
                  </div>
                )}
              </div>
              <div className="card p-5 bg-white">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-semibold text-[var(--text-primary)]">Classification Model</h3>
                  {modelStatus.riskModel && (
                    <span className="text-xs font-medium px-2 py-1 rounded bg-amber-100 text-amber-800">
                      Trained
                    </span>
                  )}
                </div>
                {modelStatus.riskModelDetails?.metrics && (
                  <div className="grid grid-cols-3 gap-2 text-center">
                    <div className="p-2 rounded-lg bg-[var(--bg-subtle)]">
                      <p className="text-[10px] uppercase tracking-wider text-[var(--text-tertiary)]">Accuracy</p>
                      <p className="text-lg font-bold text-[var(--text-primary)]">{((modelStatus.riskModelDetails.metrics.accuracy || 0) * 100).toFixed(1)}%</p>
                    </div>
                    <div className="p-2 rounded-lg bg-[var(--bg-subtle)]">
                      <p className="text-[10px] uppercase tracking-wider text-[var(--text-tertiary)]">Precision</p>
                      <p className="text-lg font-bold text-[var(--text-primary)]">{((modelStatus.riskModelDetails.metrics.precision || 0) * 100).toFixed(1)}%</p>
                    </div>
                    <div className="p-2 rounded-lg bg-[var(--bg-subtle)]">
                      <p className="text-[10px] uppercase tracking-wider text-[var(--text-tertiary)]">F1</p>
                      <p className="text-lg font-bold text-[var(--text-primary)]">{((modelStatus.riskModelDetails.metrics.f1Score || 0) * 100).toFixed(1)}%</p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Model 1: Regression */}
          <div className="card p-6 bg-white">
            <h2 className="text-xl font-semibold text-[var(--text-primary)] mb-1">
              Model 1: Player Load Prediction
            </h2>
            <p className="text-sm text-[var(--text-secondary)] mb-6">
              Regression model that predicts total player load based on session performance metrics.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-6">
              <div className="p-3 rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-subtle)]">
                <p className="text-[10px] uppercase tracking-wider text-[var(--text-tertiary)]">Type</p>
                <p className="text-sm font-semibold text-[var(--text-primary)]">Regression</p>
                <p className="text-xs text-[var(--text-tertiary)] mt-0.5">Continuous values</p>
              </div>
              <div className="p-3 rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-subtle)]">
                <p className="text-[10px] uppercase tracking-wider text-[var(--text-tertiary)]">Algorithm</p>
                <p className="text-sm font-semibold text-[var(--text-primary)]">Gradient Boosting</p>
                <p className="text-xs text-[var(--text-tertiary)] mt-0.5">Ensemble learning</p>
              </div>
              <div className="p-3 rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-subtle)]">
                <p className="text-[10px] uppercase tracking-wider text-[var(--text-tertiary)]">Objective</p>
                <p className="text-sm font-semibold text-[var(--text-primary)]">Predict Load</p>
                <p className="text-xs text-[var(--text-tertiary)] mt-0.5">Total player load</p>
              </div>
            </div>

            <h3 className="text-sm font-semibold text-[var(--text-primary)] mb-3">
              Feature Importance (Top 8)
            </h3>
            <div className="rounded-lg border border-[var(--border-subtle)] p-4 mb-6 bg-[var(--bg-subtle)]">
              <div className="h-64 min-h-[256px] w-full">
                <ResponsiveContainer width="100%" height={256} minWidth={0}>
                        <BarChart 
                          data={loadFeatureChartData} 
                          layout="vertical"
                          margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" stroke="#d4c5b0" horizontal={false} />
                          <XAxis 
                            type="number" 
                            stroke="#8b7355"
                            fontSize={11}
                            tickLine={false}
                            axisLine={false}
                            domain={[0, 100]}
                          />
                          <YAxis 
                            dataKey="name" 
                            type="category"
                            stroke="#8b7355"
                            fontSize={11}
                            tickLine={false}
                            axisLine={false}
                            width={95}
                          />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: '#ffffff',
                              border: '2px solid #d4c5b0',
                              borderRadius: '4px',
                              boxShadow: '2px 2px 6px rgba(44, 36, 22, 0.15)',
                              fontFamily: 'Georgia, serif'
                            }}
                            formatter={(value: any) => [`${value}%`, 'Importance']}
                            labelStyle={{ color: '#2c2416', fontWeight: 'bold' }}
                          />
                          <Bar 
                            dataKey="importance" 
                            radius={[0, 6, 6, 0]}
                          >
                            {loadFeatureChartData.map((entry, index) => {
                              const colors = ['#2d5016', '#4a7c2a', '#5a8f35', '#6ba045'];
                              const colorIndex = entry.category === 'Critical' || entry.category === 'Very High' ? 0 : 
                                                entry.category === 'High' ? 1 : 2;
                              return <Cell key={`cell-${index}`} fill={colors[colorIndex]} />;
                            })}
                          </Bar>
                        </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <h3 className="text-sm font-semibold text-[var(--text-primary)] mb-3">
              All Features ({loadFeatures.length})
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-6">
              {loadFeatures.map((feature, idx) => (
                <div key={idx} className="p-3 rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-subtle)]">
                  <div className="flex items-start justify-between gap-2 mb-1">
                    <h4 className="font-semibold text-[var(--text-primary)] text-sm">{feature.name}</h4>
                    <span className={`text-[10px] px-2 py-0.5 rounded font-medium ${
                      feature.importance === 'Very High' ? 'bg-red-100 text-red-800' :
                      feature.importance === 'High' ? 'bg-amber-100 text-amber-800' :
                      'bg-[var(--bg-elevated)] text-[var(--text-tertiary)]'
                    }`}>
                      {feature.importance}
                    </span>
                  </div>
                  <p className="text-xs text-[var(--text-tertiary)]">{feature.description}</p>
                  <p className="text-xs text-[var(--text-secondary)] mt-2">
                    <span className="text-[var(--text-tertiary)]">Reason:</span> {feature.reason}
                  </p>
                </div>
              ))}
            </div>

            <div className="p-4 rounded-lg bg-[var(--accent-performance-muted)] border border-[var(--accent-performance)]/20">
              <h4 className="text-sm font-semibold text-[var(--text-primary)] mb-2">How It Works</h4>
              <ul className="text-sm text-[var(--text-secondary)] space-y-1.5 list-disc list-inside">
                <li><strong>Gradient Boosting</strong> combines multiple decision trees to create a strong predictor.</li>
                <li>Features like <strong>Sprint Distance</strong> and <strong>Work Ratio</strong> have high importance due to direct correlation with effort intensity.</li>
                <li><strong>StandardScaler</strong> normalizes features for equal weighting.</li>
                <li>The model captures <strong>non-linear relationships</strong> between metrics and total load.</li>
                <li>A high <strong>R² Score</strong> indicates the model explains a large proportion of variance in player load.</li>
              </ul>
            </div>
          </div>

          {/* Model 2: Classification */}
          <div className="card p-6 bg-white">
            <h2 className="text-xl font-semibold text-[var(--text-primary)] mb-1">
              Model 2: Injury Risk Classification
            </h2>
            <p className="text-sm text-[var(--text-secondary)] mb-6">
              Multi-class classification model that categorizes players into Low, Medium, and High injury risk levels.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-6">
              <div className="p-3 rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-subtle)]">
                <p className="text-[10px] uppercase tracking-wider text-[var(--text-tertiary)]">Type</p>
                <p className="text-sm font-semibold text-[var(--text-primary)]">Classification</p>
                <p className="text-xs text-[var(--text-tertiary)] mt-0.5">3 classes: Low, Medium, High</p>
              </div>
              <div className="p-3 rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-subtle)]">
                <p className="text-[10px] uppercase tracking-wider text-[var(--text-tertiary)]">Algorithm</p>
                <p className="text-sm font-semibold text-[var(--text-primary)]">LightGBM</p>
                <p className="text-xs text-[var(--text-tertiary)] mt-0.5">Tree-based classifier</p>
              </div>
              <div className="p-3 rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-subtle)]">
                <p className="text-[10px] uppercase tracking-wider text-[var(--text-tertiary)]">Labels</p>
                <p className="text-sm font-semibold text-[var(--text-primary)]">Quartiles</p>
                <p className="text-xs text-[var(--text-tertiary)] mt-0.5">Q25, Q75 of Player Load</p>
              </div>
            </div>

            <h3 className="text-sm font-semibold text-[var(--text-primary)] mb-3">
              Feature Importance
            </h3>
            <div className="rounded-lg border border-[var(--border-subtle)] p-4 mb-6 bg-[var(--bg-subtle)]">
              <div className="h-48 min-h-[192px] w-full">
                <ResponsiveContainer width="100%" height={192} minWidth={0}>
                        <BarChart 
                          data={riskFeatureChartData} 
                          layout="vertical"
                          margin={{ top: 5, right: 30, left: 120, bottom: 5 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" stroke="#d4c5b0" horizontal={false} />
                          <XAxis 
                            type="number" 
                            stroke="#8b7355"
                            fontSize={11}
                            tickLine={false}
                            axisLine={false}
                            domain={[0, 100]}
                          />
                          <YAxis 
                            dataKey="name" 
                            type="category"
                            stroke="#8b7355"
                            fontSize={11}
                            tickLine={false}
                            axisLine={false}
                            width={115}
                          />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: '#ffffff',
                              border: '2px solid #d4c5b0',
                              borderRadius: '4px',
                              boxShadow: '2px 2px 6px rgba(44, 36, 22, 0.15)',
                              fontFamily: 'Georgia, serif'
                            }}
                            formatter={(value: any) => [`${value}%`, 'Importance']}
                            labelStyle={{ color: '#2c2416', fontWeight: 'bold' }}
                          />
                          <Bar 
                            dataKey="importance" 
                            radius={[0, 6, 6, 0]}
                          >
                            {riskFeatureChartData.map((entry, index) => {
                              const colors = ['#dc2626', '#ffd700', '#f97316', '#8b7355'];
                              const colorIndex = entry.category === 'Critical' ? 0 : 
                                                entry.category === 'Very High' ? 1 : 
                                                entry.category === 'High' ? 2 : 3;
                              return <Cell key={`cell-${index}`} fill={colors[colorIndex]} />;
                            })}
                          </Bar>
                        </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <h3 className="text-sm font-semibold text-[var(--text-primary)] mb-3">
              Features Used ({riskFeatures.length})
            </h3>
            <div className="space-y-3 mb-6">
              {riskFeatures.map((feature, idx) => (
                <div key={idx} className="p-3 rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-subtle)]">
                  <div className="flex items-start justify-between gap-2 mb-1">
                    <h4 className="font-semibold text-[var(--text-primary)] text-sm">{feature.name}</h4>
                    <span className={`text-[10px] px-2 py-0.5 rounded font-medium ${
                      feature.importance === 'Critical' ? 'bg-red-100 text-red-800' :
                      feature.importance === 'Very High' ? 'bg-amber-100 text-amber-800' :
                      feature.importance === 'High' ? 'bg-orange-100 text-orange-800' :
                      'bg-[var(--bg-elevated)] text-[var(--text-tertiary)]'
                    }`}>
                      {feature.importance}
                    </span>
                  </div>
                  <p className="text-xs text-[var(--text-tertiary)]">{feature.description}</p>
                  <p className="text-xs text-[var(--text-secondary)] mt-2">
                    <span className="text-[var(--text-tertiary)]">Reason:</span> {feature.reason}
                  </p>
                </div>
              ))}
            </div>

            <div className="p-4 rounded-lg bg-amber-50 border border-amber-200">
              <h4 className="text-sm font-semibold text-[var(--text-primary)] mb-2">How It Works</h4>
              <ul className="text-sm text-[var(--text-secondary)] space-y-1.5 list-disc list-inside">
                <li><strong>LightGBM</strong> handles non-linear data and complex feature relationships effectively.</li>
                <li>Classes use <strong>Player Load quartiles</strong> (Q25, Q75) to divide into Low, Medium, and High risk groups.</li>
                <li><strong>Player Load</strong> is the most critical feature, directly proportional to injury risk.</li>
                <li><strong>Work Ratio</strong> and <strong>Sprint Distance</strong> indicate accumulated fatigue and repeated maximum efforts.</li>
                <li><strong>Stratified sampling</strong> ensures proportional class representation during training.</li>
                <li>High precision and recall enable accurate identification of at-risk players for preventive interventions.</li>
              </ul>
            </div>
          </div>

          {/* Pipeline */}
          <div className="card p-6 bg-white">
            <h3 className="text-sm font-semibold text-[var(--text-primary)] mb-4">
              Processing Pipeline
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <div className="p-3 rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-subtle)]">
                <p className="text-xs font-semibold text-[var(--text-primary)] mb-1">1. Preprocessing</p>
                <p className="text-sm text-[var(--text-secondary)]">
                  Numerical features normalized using <strong>StandardScaler</strong> (mean 0, std 1) for equal feature weighting.
                </p>
              </div>
              <div className="p-3 rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-subtle)]">
                <p className="text-xs font-semibold text-[var(--text-primary)] mb-1">2. Training</p>
                <p className="text-sm text-[var(--text-secondary)]">
                  Data split 80/20 (train/test) using <strong>train_test_split</strong> to prevent overfitting.
                </p>
              </div>
              <div className="p-3 rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-subtle)]">
                <p className="text-xs font-semibold text-[var(--text-primary)] mb-1">3. Cross-Validation</p>
                <p className="text-sm text-[var(--text-secondary)]">
                  <strong>5-fold cross-validation</strong> provides robust performance estimates with reduced variance.
                </p>
              </div>
              <div className="p-3 rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-subtle)]">
                <p className="text-xs font-semibold text-[var(--text-primary)] mb-1">4. Evaluation</p>
                <p className="text-sm text-[var(--text-secondary)]">
                  Regression: <strong>R², MAE, RMSE</strong>. Classification: <strong>Accuracy, Precision, Recall, F1</strong>.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'training' && (
        <div className="space-y-6">
          {!dataStatus?.loaded ? (
            <div className="card p-8 text-center bg-white">
              <h2 className="text-xl font-semibold text-[var(--text-primary)] mb-2">No Data</h2>
              <p className="text-[var(--text-secondary)] text-sm mb-6">
                Upload a CSV in Dashboard (men's or women's team) to train models.
              </p>
              <a
                href="/"
                className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-[var(--accent-performance)] text-white text-sm font-medium hover:opacity-90"
              >
                Go to Dashboard
                <ChevronRight className="w-4 h-4" />
              </a>
            </div>
          ) : (
            <>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="card p-5 bg-white flex items-center gap-4">
                  {trainingStatus?.loadModel ? (
                    <CheckCircle className="w-8 h-8 text-[var(--accent-performance)] flex-shrink-0" />
                  ) : (
                    <XCircle className="w-8 h-8 text-[var(--text-tertiary)] flex-shrink-0" />
                  )}
                  <div className="min-w-0">
                    <p className="font-semibold text-[var(--text-primary)]">Player Load Model</p>
                    <p className="text-sm text-[var(--text-secondary)] truncate">
                      {trainingStatus?.loadModel
                        ? (trainingStatus.loadModelDetails?.algorithm || 'GradientBoostingRegressor')
                        : 'Not trained yet'}
                    </p>
                    {trainingStatus?.loadModelDetails?.metrics && (
                      <p className="text-xs text-[var(--text-tertiary)] mt-1">
                        R² = {trainingStatus.loadModelDetails.metrics.r2Score ?? trainingStatus.loadModelDetails.metrics.R2 ?? 'N/A'}
                      </p>
                    )}
                  </div>
                </div>
                <div className="card p-5 bg-white flex items-center gap-4">
                  {trainingStatus?.riskModel ? (
                    <CheckCircle className="w-8 h-8 text-amber-600 flex-shrink-0" />
                  ) : (
                    <XCircle className="w-8 h-8 text-[var(--text-tertiary)] flex-shrink-0" />
                  )}
                  <div className="min-w-0">
                    <p className="font-semibold text-[var(--text-primary)]">Injury Risk Model</p>
                    <p className="text-sm text-[var(--text-secondary)] truncate">
                      {trainingStatus?.riskModel
                        ? (trainingStatus.riskModelDetails?.algorithm || 'LGBMClassifier')
                        : 'Not trained yet'}
                    </p>
                    {trainingStatus?.riskModelDetails?.metrics && (
                      <p className="text-xs text-[var(--text-tertiary)] mt-1">
                        Accuracy = {trainingStatus.riskModelDetails.metrics.accuracy ?? trainingStatus.riskModelDetails.metrics.Accuracy ?? 'N/A'}
                      </p>
                    )}
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="card p-6 bg-white">
                  <h2 className="text-lg font-semibold text-[var(--text-primary)] mb-1">Player Load Prediction</h2>
                  <p className="text-sm text-[var(--text-tertiary)] mb-4">GradientBoostingRegressor</p>
                  <p className="text-sm text-[var(--text-secondary)] mb-5">
                    Predicts Player Load based on metrics like duration, distance, speed, and accelerations.
                  </p>

                  <button
                    onClick={() => trainLoadMutation.mutate()}
                    disabled={trainLoadMutation.isPending}
                    className="w-full py-3 rounded-lg bg-[var(--accent-performance)] text-white font-medium text-sm flex items-center justify-center gap-2 disabled:opacity-50"
                  >
                    {trainLoadMutation.isPending ? (
                      <>
                        <Loader2 className="w-4 h-4 animate-spin" />
                        Training...
                      </>
                    ) : (
                      <>
                        <RefreshCw className="w-4 h-4" />
                        Retrain Model
                      </>
                    )}
                  </button>

                  {trainLoadMutation.data && (
                    <div className="mt-4 p-4 rounded-lg bg-[var(--accent-performance-muted)] border border-[var(--accent-performance)]/20">
                      <p className="text-sm font-semibold text-[var(--accent-performance)] mb-3">Training complete</p>
                      <div className="grid grid-cols-2 gap-2">
                        <div className="p-2 rounded bg-white/80">
                          <p className="text-[10px] uppercase text-[var(--text-tertiary)]">R²</p>
                          <p className="text-base font-bold text-[var(--text-primary)]">{trainLoadMutation.data.metrics.r2Score?.toFixed(3)}</p>
                        </div>
                        <div className="p-2 rounded bg-white/80">
                          <p className="text-[10px] uppercase text-[var(--text-tertiary)]">MAE</p>
                          <p className="text-base font-bold text-[var(--text-primary)]">{trainLoadMutation.data.metrics.mae?.toFixed(2)}</p>
                        </div>
                        <div className="p-2 rounded bg-white/80">
                          <p className="text-[10px] uppercase text-[var(--text-tertiary)]">RMSE</p>
                          <p className="text-base font-bold text-[var(--text-primary)]">{trainLoadMutation.data.metrics.rmse?.toFixed(2)}</p>
                        </div>
                        <div className="p-2 rounded bg-white/80">
                          <p className="text-[10px] uppercase text-[var(--text-tertiary)]">Time</p>
                          <p className="text-base font-bold text-[var(--text-primary)]">{trainLoadMutation.data.trainingTime}s</p>
                        </div>
                      </div>
                    </div>
                  )}

                  {trainLoadMutation.isError && (
                    <div className="mt-4 p-3 rounded-lg bg-red-50 border border-red-200 text-sm text-red-700">
                      Training failed: {(trainLoadMutation.error as Error)?.message || 'Unknown error'}
                    </div>
                  )}
                </div>

                <div className="card p-6 bg-white">
                  <h2 className="text-lg font-semibold text-[var(--text-primary)] mb-1">Injury Risk Classification</h2>
                  <p className="text-sm text-[var(--text-tertiary)] mb-4">LGBMClassifier (LightGBM)</p>
                  <p className="text-sm text-[var(--text-secondary)] mb-5">
                    Classifies players into Low, Medium, or High injury risk using LightGBM.
                  </p>

                  <button
                    onClick={() => trainRiskMutation.mutate()}
                    disabled={trainRiskMutation.isPending}
                    className="w-full py-3 rounded-lg bg-amber-600 hover:bg-amber-700 text-white font-medium text-sm flex items-center justify-center gap-2 disabled:opacity-50"
                  >
                    {trainRiskMutation.isPending ? (
                      <>
                        <Loader2 className="w-4 h-4 animate-spin" />
                        Training...
                      </>
                    ) : (
                      <>
                        <RefreshCw className="w-4 h-4" />
                        Retrain Model
                      </>
                    )}
                  </button>

                  {trainRiskMutation.data && (
                    <div className="mt-4 p-4 rounded-lg bg-amber-50 border border-amber-200">
                      <p className="text-sm font-semibold text-amber-800 mb-3">Training complete</p>
                      <div className="grid grid-cols-2 gap-2">
                        <div className="p-2 rounded bg-white/80">
                          <p className="text-[10px] uppercase text-[var(--text-tertiary)]">Accuracy</p>
                          <p className="text-base font-bold text-[var(--text-primary)]">{((trainRiskMutation.data.metrics.accuracy || 0) * 100).toFixed(1)}%</p>
                        </div>
                        <div className="p-2 rounded bg-white/80">
                          <p className="text-[10px] uppercase text-[var(--text-tertiary)]">Precision</p>
                          <p className="text-base font-bold text-[var(--text-primary)]">{((trainRiskMutation.data.metrics.precision || 0) * 100).toFixed(1)}%</p>
                        </div>
                        <div className="p-2 rounded bg-white/80">
                          <p className="text-[10px] uppercase text-[var(--text-tertiary)]">Recall</p>
                          <p className="text-base font-bold text-[var(--text-primary)]">{((trainRiskMutation.data.metrics.recall || 0) * 100).toFixed(1)}%</p>
                        </div>
                        <div className="p-2 rounded bg-white/80">
                          <p className="text-[10px] uppercase text-[var(--text-tertiary)]">F1</p>
                          <p className="text-base font-bold text-[var(--text-primary)]">{((trainRiskMutation.data.metrics.f1Score || 0) * 100).toFixed(1)}%</p>
                        </div>
                      </div>
                    </div>
                  )}

                  {trainRiskMutation.isError && (
                    <div className="mt-4 p-3 rounded-lg bg-red-50 border border-red-200 text-sm text-red-700">
                      Training failed: {(trainRiskMutation.error as Error)?.message || 'Unknown error'}
                    </div>
                  )}
                </div>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
