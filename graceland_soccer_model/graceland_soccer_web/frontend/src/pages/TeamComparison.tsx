import { useQuery } from '@tanstack/react-query';
import { useTeam } from '../contexts/TeamContext';
import { dashboardApi, playersApi } from '../services/api';
import { BarChart3, TrendingUp, Users, Zap, Activity, Gauge, Award } from 'lucide-react';

export default function TeamComparison() {
  const { teamStatus, currentTeam } = useTeam();

  // Get current team's data
  const { data: currentKPIs } = useQuery({
    queryKey: ['dashboard', 'kpis', currentTeam],
    queryFn: dashboardApi.getKPIs,
    enabled: (teamStatus?.mens?.loaded || teamStatus?.womens?.loaded) || false,
  });

  const { data: currentPlayers } = useQuery({
    queryKey: ['players', currentTeam],
    queryFn: playersApi.getAll,
    enabled: (teamStatus?.mens?.loaded || teamStatus?.womens?.loaded) || false,
  });

  // Determine which data to show based on loaded teams and current team
  const mensKPIs = (currentTeam === 'mens' && teamStatus?.mens?.loaded) ? currentKPIs : null;
  const womensKPIs = (currentTeam === 'womens' && teamStatus?.womens?.loaded) ? currentKPIs : null;
  const mensPlayers = (currentTeam === 'mens' && teamStatus?.mens?.loaded) ? currentPlayers : null;
  const womensPlayers = (currentTeam === 'womens' && teamStatus?.womens?.loaded) ? currentPlayers : null;
  
  // Check if current team has data loaded
  const showCurrentTeamData = (currentTeam === 'mens' && teamStatus?.mens?.loaded) || 
                              (currentTeam === 'womens' && teamStatus?.womens?.loaded);
  
  // Use current KPIs for the active team
  const activeKPIs = showCurrentTeamData ? currentKPIs : null;

  // Build metrics based on available data
  const comparisonMetrics = [
    {
      name: 'Total Players',
      mensValue: (currentTeam === 'mens' && activeKPIs) ? activeKPIs.totalPlayers : (mensKPIs?.totalPlayers || 0),
      womensValue: (currentTeam === 'womens' && activeKPIs) ? activeKPIs.totalPlayers : (womensKPIs?.totalPlayers || 0),
      icon: Users,
      color: 'text-blue-400',
    },
    {
      name: 'Average Team Load',
      mensValue: (currentTeam === 'mens' && activeKPIs) ? activeKPIs.avgTeamLoad : (mensKPIs?.avgTeamLoad || 0),
      womensValue: (currentTeam === 'womens' && activeKPIs) ? activeKPIs.avgTeamLoad : (womensKPIs?.avgTeamLoad || 0),
      icon: Zap,
      color: 'text-yellow-400',
      unit: 'units',
    },
    {
      name: 'High Risk Players',
      mensValue: (currentTeam === 'mens' && activeKPIs) ? activeKPIs.highRiskPlayers : (mensKPIs?.highRiskPlayers || 0),
      womensValue: (currentTeam === 'womens' && activeKPIs) ? activeKPIs.highRiskPlayers : (womensKPIs?.highRiskPlayers || 0),
      icon: Activity,
      color: 'text-red-400',
    },
    {
      name: 'Average Team Speed',
      mensValue: (currentTeam === 'mens' && activeKPIs) ? activeKPIs.avgTeamSpeed : (mensKPIs?.avgTeamSpeed || 0),
      womensValue: (currentTeam === 'womens' && activeKPIs) ? activeKPIs.avgTeamSpeed : (womensKPIs?.avgTeamSpeed || 0),
      icon: Gauge,
      color: 'text-green-400',
      unit: 'mph',
    },
  ];

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-slate-900 to-slate-800 rounded-xl p-6 border border-slate-700">
        <div className="flex items-center gap-3 mb-2">
          <div className="p-3 bg-purple-600/20 rounded-lg">
            <BarChart3 className="w-6 h-6 text-purple-400" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-white">Team Comparison</h1>
            <p className="text-slate-400">
              {teamStatus?.mens?.loaded && teamStatus?.womens?.loaded 
                ? "Compare Men's and Women's team performance"
                : teamStatus?.mens?.loaded 
                  ? "Men's team performance metrics"
                  : teamStatus?.womens?.loaded
                    ? "Women's team performance metrics"
                    : "Team performance overview"}
            </p>
          </div>
        </div>
      </div>

      {/* Team Status */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className={`p-4 rounded-lg border ${teamStatus?.mens?.loaded ? 'bg-blue-900/20 border-blue-700/50' : 'bg-slate-800/50 border-slate-700'}`}>
          <div className="flex items-center gap-2 mb-2">
            <Users className={`w-5 h-5 ${teamStatus?.mens?.loaded ? 'text-blue-400' : 'text-slate-500'}`} />
            <h3 className="font-semibold text-white">Men's Team</h3>
            {teamStatus?.mens?.loaded && <span className="text-xs bg-blue-600 text-white px-2 py-1 rounded">Loaded</span>}
          </div>
          {teamStatus?.mens?.loaded ? (
            <div className="text-sm text-slate-300">
              <p>Players: <span className="text-blue-400 font-semibold">{teamStatus.mens.rowCount}</span></p>
            </div>
          ) : (
            <p className="text-sm text-slate-500">No data loaded</p>
          )}
        </div>
        <div className={`p-4 rounded-lg border ${teamStatus?.womens?.loaded ? 'bg-pink-900/20 border-pink-700/50' : 'bg-slate-800/50 border-slate-700'}`}>
          <div className="flex items-center gap-2 mb-2">
            <Users className={`w-5 h-5 ${teamStatus?.womens?.loaded ? 'text-pink-400' : 'text-slate-500'}`} />
            <h3 className="font-semibold text-white">Women's Team</h3>
            {teamStatus?.womens?.loaded && <span className="text-xs bg-pink-600 text-white px-2 py-1 rounded">Loaded</span>}
          </div>
          {teamStatus?.womens?.loaded ? (
            <div className="text-sm text-slate-300">
              <p>Players: <span className="text-pink-400 font-semibold">{teamStatus.womens.rowCount}</span></p>
            </div>
          ) : (
            <p className="text-sm text-slate-500">No data loaded</p>
          )}
        </div>
      </div>

      {/* Comparison Metrics */}
      {showCurrentTeamData ? (
        <div className="bg-slate-900/50 rounded-xl p-6 border border-slate-700">
          <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-purple-400" />
            {teamStatus?.mens?.loaded && teamStatus?.womens?.loaded 
              ? 'Performance Comparison' 
              : 'Team Performance Metrics'}
          </h2>
          <div className="space-y-4">
            {comparisonMetrics.map((metric, idx) => {
              const Icon = metric.icon;
              const hasBothTeams = teamStatus?.mens?.loaded && teamStatus?.womens?.loaded;
              const difference = hasBothTeams ? metric.mensValue - metric.womensValue : 0;
              const percentageDiff = hasBothTeams && metric.womensValue > 0 
                ? ((difference / metric.womensValue) * 100).toFixed(1)
                : '0.0';
              
              return (
                <div key={idx} className="bg-slate-800/30 p-4 rounded-lg border border-slate-700/50">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <Icon className={`w-5 h-5 ${metric.color}`} />
                      <h3 className="font-semibold text-white">{metric.name}</h3>
                    </div>
                  </div>
                  <div className={`grid gap-4 ${hasBothTeams ? 'grid-cols-2' : 'grid-cols-1'}`}>
                    {(currentTeam === 'mens' && teamStatus?.mens?.loaded && activeKPIs) && (
                      <div className="text-center p-3 bg-blue-900/20 rounded-lg">
                        <p className="text-xs text-slate-400 mb-1">Men's</p>
                        <p className="text-2xl font-bold text-blue-400">
                          {typeof metric.mensValue === 'number' ? metric.mensValue.toFixed(1) : metric.mensValue}
                          {metric.unit && <span className="text-sm ml-1">{metric.unit}</span>}
                        </p>
                      </div>
                    )}
                    {(currentTeam === 'womens' && teamStatus?.womens?.loaded && activeKPIs) && (
                      <div className="text-center p-3 bg-pink-900/20 rounded-lg">
                        <p className="text-xs text-slate-400 mb-1">Women's</p>
                        <p className="text-2xl font-bold text-pink-400">
                          {typeof metric.womensValue === 'number' ? metric.womensValue.toFixed(1) : metric.womensValue}
                          {metric.unit && <span className="text-sm ml-1">{metric.unit}</span>}
                        </p>
                      </div>
                    )}
                  </div>
                  {hasBothTeams && (
                    <div className="mt-3 pt-3 border-t border-slate-700">
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-slate-400">Difference</span>
                        <span className={`font-semibold ${difference >= 0 ? 'text-blue-400' : 'text-pink-400'}`}>
                          {difference >= 0 ? '+' : ''}{typeof difference === 'number' ? difference.toFixed(1) : difference}
                          {metric.unit && <span className="ml-1">{metric.unit}</span>}
                          <span className="ml-2 text-slate-500">({percentageDiff}%)</span>
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      ) : (
        <div className="bg-slate-900/50 rounded-xl p-12 border border-slate-700 text-center">
          <Award className="w-12 h-12 text-slate-600 mx-auto mb-4" />
          <p className="text-slate-400 mb-2">No team data loaded</p>
          <p className="text-slate-500 text-sm">Upload CSV files for Men's or Women's team to view performance metrics</p>
        </div>
      )}

      {/* Additional Insights */}
      {showCurrentTeamData && currentPlayers && (
        <div className="grid grid-cols-1 gap-4">
          {currentTeam === 'mens' && mensPlayers && (
            <div className="bg-slate-900/50 rounded-xl p-6 border border-slate-700">
              <h3 className="font-semibold text-white mb-4">Men's Team Top Performers</h3>
              <div className="space-y-2">
                {mensPlayers.slice(0, 5).map((player, idx) => (
                  <div key={player.id} className="flex items-center justify-between p-2 bg-slate-800/30 rounded-lg">
                    <div className="flex items-center gap-2">
                      <span className="text-slate-500 text-sm">#{idx + 1}</span>
                      <span className="text-white text-sm">{player.name}</span>
                    </div>
                    <span className="text-blue-400 font-semibold">{player.avgLoad}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
          {currentTeam === 'womens' && womensPlayers && (
            <div className="bg-slate-900/50 rounded-xl p-6 border border-slate-700">
              <h3 className="font-semibold text-white mb-4">Women's Team Top Performers</h3>
              <div className="space-y-2">
                {womensPlayers.slice(0, 5).map((player, idx) => (
                  <div key={player.id} className="flex items-center justify-between p-2 bg-slate-800/30 rounded-lg">
                    <div className="flex items-center gap-2">
                      <span className="text-slate-500 text-sm">#{idx + 1}</span>
                      <span className="text-white text-sm">{player.name}</span>
                    </div>
                    <span className="text-pink-400 font-semibold">{player.avgLoad}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
