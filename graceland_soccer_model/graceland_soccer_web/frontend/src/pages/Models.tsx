import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Brain, TrendingUp, AlertTriangle, Target, BarChart3, Zap, Activity, Gauge, Dumbbell, CheckCircle, XCircle, Loader2, Cpu, Info, RefreshCw, ChevronRight } from 'lucide-react';
import { getModelStatus, trainingApi, useDataStatus } from '../services/api';

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
      reason: 'Longer sessions generate greater accumulated load on the player'
    },
    {
      name: 'Distance (miles)',
      description: 'Total distance covered during the session',
      importance: 'High',
      reason: 'Greater distance implies greater physical effort and metabolic load'
    },
    {
      name: 'Sprint Distance (yards)',
      description: 'Distance covered in high-intensity sprints',
      importance: 'Very High',
      reason: 'Sprints generate significant neuromuscular load and fatigue'
    },
    {
      name: 'Top Speed (mph)',
      description: 'Maximum speed reached',
      importance: 'High',
      reason: 'Maximum speeds indicate maximum efforts that increase load'
    },
    {
      name: 'Max Acceleration (yd/s/s)',
      description: 'Maximum acceleration during the session',
      importance: 'High',
      reason: 'Maximum accelerations generate high muscular and metabolic load'
    },
    {
      name: 'Max Deceleration (yd/s/s)',
      description: 'Maximum deceleration during the session',
      importance: 'Medium',
      reason: 'Decelerations generate eccentric load on muscles'
    },
    {
      name: 'Work Ratio',
      description: 'Work ratio (relationship between work and recovery)',
      importance: 'Very High',
      reason: 'Indicates work intensity and fatigue accumulation'
    },
    {
      name: 'Energy (kcal)',
      description: 'Total energy consumed during the session',
      importance: 'High',
      reason: 'Reflects total energy expenditure and metabolic load'
    },
    {
      name: 'Hr Load',
      description: 'Load based on heart rate',
      importance: 'Medium',
      reason: 'Indicates cardiovascular response to effort'
    },
    {
      name: 'Impacts',
      description: 'Number of impacts received',
      importance: 'Medium',
      reason: 'Repeated impacts can contribute to accumulated load'
    },
    {
      name: 'Power Plays',
      description: 'High-power plays',
      importance: 'Medium',
      reason: 'Indicates moments of maximum intensity during the session'
    },
    {
      name: 'Power Score (w/kg)',
      description: 'Power score relative to weight',
      importance: 'High',
      reason: 'Reflects power generated and neuromuscular load'
    },
    {
      name: 'Distance Per Min (yd/min)',
      description: 'Average distance per minute',
      importance: 'Medium',
      reason: 'Indicates average work pace during the session'
    }
  ];

  const riskFeatures = [
    {
      name: 'Player Load',
      description: 'Total player load (main objective)',
      importance: 'Critical',
      reason: 'It is the main metric that determines the injury risk level'
    },
    {
      name: 'Work Ratio',
      description: 'Work ratio vs recovery',
      importance: 'Very High',
      reason: 'High ratios indicate accumulated fatigue and greater overload risk'
    },
    {
      name: 'Sprint Distance',
      description: 'Distance in high-intensity sprints',
      importance: 'High',
      reason: 'Excessive sprints increase the risk of muscle injuries'
    },
    {
      name: 'Top Speed',
      description: 'Maximum speed reached',
      importance: 'High',
      reason: 'Repeated maximum speeds increase injury risk'
    },
    {
      name: 'Distance',
      description: 'Total distance covered',
      importance: 'Medium',
      reason: 'Very high distances may indicate cumulative overload'
    }
  ];

  const tabs = [
    { id: 'explanation' as ModelTab, label: 'Model Explanation', icon: Brain },
    { id: 'training' as ModelTab, label: 'Train Models', icon: Dumbbell },
  ];

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-slate-900 to-slate-800 rounded-xl p-6 border border-slate-700">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-3 bg-blue-600/20 rounded-lg">
            <Brain className="w-6 h-6 text-blue-400" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-white">Machine Learning Models</h1>
            <p className="text-slate-400">Technical explanation and model training</p>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex gap-2 border-t border-slate-700 pt-4 mt-4">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`
                  flex items-center gap-2 px-4 py-2 rounded-lg transition-colors
                  ${activeTab === tab.id
                    ? 'bg-slate-700/50 text-white border border-slate-600'
                    : 'text-slate-400 hover:text-slate-300 hover:bg-slate-800/50'
                  }
                `}
              >
                <Icon className="w-4 h-4" />
                <span className="font-medium">{tab.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Tab Content */}
      {activeTab === 'explanation' && (
        <>

      {/* Model Status */}
      {!isLoading && modelStatus && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className={`p-4 rounded-lg border ${modelStatus.loadModel ? 'bg-green-900/20 border-green-700/50' : 'bg-slate-800/50 border-slate-700'}`}>
            <div className="flex items-center gap-2 mb-2">
              <TrendingUp className={`w-5 h-5 ${modelStatus.loadModel ? 'text-green-400' : 'text-slate-500'}`} />
              <h3 className="font-semibold text-white">Regression Model</h3>
              {modelStatus.loadModel && <span className="text-xs bg-green-600 text-white px-2 py-1 rounded">Trained</span>}
            </div>
            {modelStatus.loadModelDetails?.metrics && (
              <div className="text-sm text-slate-300 space-y-1">
                <p>R² Score: <span className="text-green-400 font-semibold">{modelStatus.loadModelDetails.metrics.r2Score}</span></p>
                <p>MAE: <span className="text-slate-400">{modelStatus.loadModelDetails.metrics.mae}</span></p>
                <p>RMSE: <span className="text-slate-400">{modelStatus.loadModelDetails.metrics.rmse}</span></p>
              </div>
            )}
          </div>
          <div className={`p-4 rounded-lg border ${modelStatus.riskModel ? 'bg-orange-900/20 border-orange-700/50' : 'bg-slate-800/50 border-slate-700'}`}>
            <div className="flex items-center gap-2 mb-2">
              <AlertTriangle className={`w-5 h-5 ${modelStatus.riskModel ? 'text-orange-400' : 'text-slate-500'}`} />
              <h3 className="font-semibold text-white">Classification Model</h3>
              {modelStatus.riskModel && <span className="text-xs bg-orange-600 text-white px-2 py-1 rounded">Trained</span>}
            </div>
            {modelStatus.riskModelDetails?.metrics && (
              <div className="text-sm text-slate-300 space-y-1">
                <p>Accuracy: <span className="text-orange-400 font-semibold">{modelStatus.riskModelDetails.metrics.accuracy}</span></p>
                <p>Precision: <span className="text-slate-400">{modelStatus.riskModelDetails.metrics.precision}</span></p>
                <p>F1 Score: <span className="text-slate-400">{modelStatus.riskModelDetails.metrics.f1Score}</span></p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Model 1: Regression */}
      <div className="bg-slate-900/50 rounded-xl p-6 border border-slate-700">
        <div className="flex items-start gap-4 mb-6">
          <div className="p-3 bg-blue-600/20 rounded-lg">
            <TrendingUp className="w-6 h-6 text-blue-400" />
          </div>
          <div className="flex-1">
            <h2 className="text-xl font-bold text-white mb-2">Model 1: Player Load Prediction (Regression)</h2>
            <p className="text-slate-400 text-sm mb-4">
              This model predicts the total player load (Player Load) based on session performance metrics.
            </p>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
              <div className="bg-slate-800/50 p-4 rounded-lg border border-slate-700">
                <div className="flex items-center gap-2 mb-2">
                  <Target className="w-4 h-4 text-blue-400" />
                  <span className="text-sm font-semibold text-white">Type</span>
                </div>
                <p className="text-slate-300 text-sm">Regression</p>
                <p className="text-slate-500 text-xs mt-1">Predicts continuous values</p>
              </div>
              <div className="bg-slate-800/50 p-4 rounded-lg border border-slate-700">
                <div className="flex items-center gap-2 mb-2">
                  <BarChart3 className="w-4 h-4 text-blue-400" />
                  <span className="text-sm font-semibold text-white">Algorithm</span>
                </div>
                <p className="text-slate-300 text-sm">Gradient Boosting / XGBoost</p>
                <p className="text-slate-500 text-xs mt-1">Ensemble learning</p>
              </div>
              <div className="bg-slate-800/50 p-4 rounded-lg border border-slate-700">
                <div className="flex items-center gap-2 mb-2">
                  <Zap className="w-4 h-4 text-blue-400" />
                  <span className="text-sm font-semibold text-white">Objective</span>
                </div>
                <p className="text-slate-300 text-sm">Predict Player Load</p>
                <p className="text-slate-500 text-xs mt-1">Total player load</p>
              </div>
            </div>

            <div className="mb-6">
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <Activity className="w-5 h-5 text-blue-400" />
                Features Used ({loadFeatures.length})
              </h3>
              <div className="space-y-3">
                {loadFeatures.map((feature, idx) => (
                  <div key={idx} className="bg-slate-800/30 p-4 rounded-lg border border-slate-700/50">
                    <div className="flex items-start justify-between mb-2">
                      <div>
                        <h4 className="font-semibold text-white">{feature.name}</h4>
                        <p className="text-sm text-slate-400 mt-1">{feature.description}</p>
                      </div>
                      <span className={`text-xs px-2 py-1 rounded ${
                        feature.importance === 'Very High' || feature.importance === 'Critical' 
                          ? 'bg-red-900/50 text-red-300' 
                          : feature.importance === 'High' 
                          ? 'bg-orange-900/50 text-orange-300' 
                          : 'bg-blue-900/50 text-blue-300'
                      }`}>
                        {feature.importance}
                      </span>
                    </div>
                    <p className="text-sm text-slate-300 mt-2">
                      <span className="text-slate-500">Reason:</span> {feature.reason}
                    </p>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-blue-900/20 border border-blue-700/50 rounded-lg p-4">
              <h4 className="font-semibold text-blue-300 mb-2">Why do we get these results?</h4>
              <ul className="text-sm text-slate-300 space-y-2 list-disc list-inside">
                <li>The model uses <strong className="text-white">Gradient Boosting</strong>, which combines multiple weak decision trees to create a strong predictor.</li>
                <li>Features like <strong className="text-white">Sprint Distance</strong> and <strong className="text-white">Work Ratio</strong> have high importance because they are directly related to effort intensity.</li>
                <li>The <strong className="text-white">StandardScaler</strong> normalizes features, allowing the model to learn patterns independently of variable scales.</li>
                <li>The model learns non-linear relationships between performance metrics and total load, capturing complex interactions between variables.</li>
                <li>A high R² Score indicates that the model explains a large proportion of variance in player load based on session metrics.</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      {/* Model 2: Classification */}
      <div className="bg-slate-900/50 rounded-xl p-6 border border-slate-700">
        <div className="flex items-start gap-4 mb-6">
          <div className="p-3 bg-orange-600/20 rounded-lg">
            <AlertTriangle className="w-6 h-6 text-orange-400" />
          </div>
          <div className="flex-1">
            <h2 className="text-xl font-bold text-white mb-2">Model 2: Injury Risk Classification</h2>
            <p className="text-slate-400 text-sm mb-4">
              This model classifies players into three risk levels: Low, Medium, and High, based on their performance metrics.
            </p>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
              <div className="bg-slate-800/50 p-4 rounded-lg border border-slate-700">
                <div className="flex items-center gap-2 mb-2">
                  <Target className="w-4 h-4 text-orange-400" />
                  <span className="text-sm font-semibold text-white">Tipo</span>
                </div>
                <p className="text-slate-300 text-sm">Multi-class Classification</p>
                <p className="text-slate-500 text-xs mt-1">3 classes: Low, Medium, High</p>
              </div>
              <div className="bg-slate-800/50 p-4 rounded-lg border border-slate-700">
                <div className="flex items-center gap-2 mb-2">
                  <BarChart3 className="w-4 h-4 text-orange-400" />
                  <span className="text-sm font-semibold text-white">Algorithm</span>
                </div>
                <p className="text-slate-300 text-sm">LightGBM / Random Forest</p>
                <p className="text-slate-500 text-xs mt-1">Tree-based classifier</p>
              </div>
              <div className="bg-slate-800/50 p-4 rounded-lg border border-slate-700">
                <div className="flex items-center gap-2 mb-2">
                  <Gauge className="w-4 h-4 text-orange-400" />
                  <span className="text-sm font-semibold text-white">Labels</span>
                </div>
                <p className="text-slate-300 text-sm">Based on quartiles</p>
                <p className="text-slate-500 text-xs mt-1">Q25, Q75 of Player Load</p>
              </div>
            </div>

            <div className="mb-6">
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <Activity className="w-5 h-5 text-orange-400" />
                Features Used ({riskFeatures.length})
              </h3>
              <div className="space-y-3">
                {riskFeatures.map((feature, idx) => (
                  <div key={idx} className="bg-slate-800/30 p-4 rounded-lg border border-slate-700/50">
                    <div className="flex items-start justify-between mb-2">
                      <div>
                        <h4 className="font-semibold text-white">{feature.name}</h4>
                        <p className="text-sm text-slate-400 mt-1">{feature.description}</p>
                      </div>
                      <span className={`text-xs px-2 py-1 rounded ${
                        feature.importance === 'Critical' 
                          ? 'bg-red-900/50 text-red-300' 
                          : feature.importance === 'Very High' 
                          ? 'bg-orange-900/50 text-orange-300' 
                          : feature.importance === 'High' 
                          ? 'bg-yellow-900/50 text-yellow-300' 
                          : 'bg-blue-900/50 text-blue-300'
                      }`}>
                        {feature.importance}
                      </span>
                    </div>
                    <p className="text-sm text-slate-300 mt-2">
                      <span className="text-slate-500">Reason:</span> {feature.reason}
                    </p>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-orange-900/20 border border-orange-700/50 rounded-lg p-4">
              <h4 className="font-semibold text-orange-300 mb-2">Why do we get these results?</h4>
              <ul className="text-sm text-slate-300 space-y-2 list-disc list-inside">
                <li>The model uses <strong className="text-white">LightGBM</strong> or <strong className="text-white">Random Forest</strong>, algorithms that handle non-linear data and complex relationships between features well.</li>
                <li>Classes are created using <strong className="text-white">Player Load quartiles</strong> (Q25 and Q75), dividing data into three groups: low risk (0-Q25), medium risk (Q25-Q75), and high risk (Q75+).</li>
                <li>The most important feature is <strong className="text-white">Player Load</strong>, as it is directly proportional to injury risk according to scientific studies.</li>
                <li>Features like <strong className="text-white">Work Ratio</strong> and <strong className="text-white">Sprint Distance</strong> are critical because they indicate accumulated fatigue and repeated maximum efforts.</li>
                <li>The model uses <strong className="text-white">stratified sampling</strong> during training to ensure each class is proportionally represented.</li>
                <li>High precision and recall indicate that the model can correctly identify at-risk players, enabling preventive interventions.</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      {/* Pipeline Explanation */}
      <div className="bg-slate-900/50 rounded-xl p-6 border border-slate-700">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Zap className="w-5 h-5 text-purple-400" />
          Processing Pipeline
        </h3>
        <div className="space-y-4">
          <div className="bg-slate-800/30 p-4 rounded-lg border border-slate-700/50">
            <h4 className="font-semibold text-white mb-2">1. Preprocessing</h4>
            <p className="text-sm text-slate-300">
              Numerical features are normalized using <strong className="text-white">StandardScaler</strong>, 
              which transforms data to a distribution with mean 0 and standard deviation 1. This allows 
              all features to have the same weight during training.
            </p>
          </div>
          <div className="bg-slate-800/30 p-4 rounded-lg border border-slate-700/50">
            <h4 className="font-semibold text-white mb-2">2. Training</h4>
            <p className="text-sm text-slate-300">
              Data is split into training (80%) and test (20%) sets using <strong className="text-white">train_test_split</strong>. 
              The model is trained on the training set and evaluated on the test set to 
              avoid overfitting.
            </p>
          </div>
          <div className="bg-slate-800/30 p-4 rounded-lg border border-slate-700/50">
            <h4 className="font-semibold text-white mb-2">3. Cross-Validation</h4>
            <p className="text-sm text-slate-300">
              <strong className="text-white">Cross-validation (5 folds)</strong> is used to obtain a more robust 
              estimate of model performance, reducing variance in evaluation metrics.
            </p>
          </div>
          <div className="bg-slate-800/30 p-4 rounded-lg border border-slate-700/50">
            <h4 className="font-semibold text-white mb-2">4. Evaluation</h4>
            <p className="text-sm text-slate-300">
              For regression: <strong className="text-white">R² Score</strong> (coefficient of determination), 
              <strong className="text-white"> MAE</strong> (mean absolute error), and <strong className="text-white">RMSE</strong> (root mean squared error). 
              For classification: <strong className="text-white">Accuracy</strong>, <strong className="text-white">Precision</strong>, 
              <strong className="text-white"> Recall</strong>, and <strong className="text-white">F1 Score</strong>.
            </p>
          </div>
        </div>
      </div>
        </>
      )}

      {activeTab === 'training' && (
        <div className="space-y-6">
          {!dataStatus?.loaded ? (
            <div className="card p-8 text-center">
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
          ) : (
            <>
              {/* Model Status Cards */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="card p-5">
                  <div className="flex items-center gap-4">
                    <div className={`
                      w-14 h-14 rounded-xl flex items-center justify-center
                      ${trainingStatus?.loadModel 
                        ? 'bg-slate-800/60 border border-slate-600' 
                        : 'bg-slate-800 border border-slate-700'
                      }
                    `}>
                      {trainingStatus?.loadModel ? (
                        <CheckCircle className="w-7 h-7 text-white" />
                      ) : (
                        <XCircle className="w-7 h-7 text-slate-500" />
                      )}
                    </div>
                    <div className="flex-1">
                      <p className="font-semibold text-white">Player Load Model</p>
                      <p className="text-sm text-slate-500">
                        {trainingStatus?.loadModel ? (
                          <span className="text-emerald-400">
                            {trainingStatus.loadModelDetails?.algorithm || 'GradientBoostingRegressor'}
                          </span>
                        ) : (
                          'Not trained yet'
                        )}
                      </p>
                      {trainingStatus?.loadModelDetails?.metrics && (
                        <p className="text-xs text-slate-500 mt-1">
                          R² = {trainingStatus.loadModelDetails.metrics.r2Score || trainingStatus.loadModelDetails.metrics.R2 || 'N/A'}
                        </p>
                      )}
                    </div>
                  </div>
                </div>

                <div className="card p-5">
                  <div className="flex items-center gap-4">
                    <div className={`
                      w-14 h-14 rounded-xl flex items-center justify-center
                      ${trainingStatus?.riskModel 
                        ? 'bg-slate-800/60 border border-slate-600' 
                        : 'bg-slate-800 border border-slate-700'
                      }
                    `}>
                      {trainingStatus?.riskModel ? (
                        <CheckCircle className="w-7 h-7 text-white" />
                      ) : (
                        <XCircle className="w-7 h-7 text-slate-500" />
                      )}
                    </div>
                    <div className="flex-1">
                      <p className="font-semibold text-white">Injury Risk Model</p>
                      <p className="text-sm text-slate-500">
                        {trainingStatus?.riskModel ? (
                          <span className="text-emerald-400">
                            {trainingStatus.riskModelDetails?.algorithm || 'LGBMClassifier'}
                          </span>
                        ) : (
                          'Not trained yet'
                        )}
                      </p>
                      {trainingStatus?.riskModelDetails?.metrics && (
                        <p className="text-xs text-slate-500 mt-1">
                          Accuracy = {trainingStatus.riskModelDetails.metrics.accuracy || trainingStatus.riskModelDetails.metrics.Accuracy || 'N/A'}
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
            </>
          )}
        </div>
      )}
    </div>
  );
}
