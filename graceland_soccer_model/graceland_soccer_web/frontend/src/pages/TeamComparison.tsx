import { useQuery } from '@tanstack/react-query';
import { useTeam } from '../contexts/TeamContext';
import { dashboardApi, playersApi } from '../services/api';
import { BarChart3, TrendingUp, Users, Zap, Activity, Gauge } from 'lucide-react';

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
  
  const bothTeamsLoaded = Boolean(teamStatus?.mens?.loaded && teamStatus?.womens?.loaded);

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
      color: 'text-[#1e40af]',
      unit: 'mph',
    },
  ];

  if (!bothTeamsLoaded) {
    return (
      <div className="flex items-center justify-center min-h-[50vh]">
        <div className="panel panel--elevated p-12 text-center max-w-xl">
          <BarChart3 className="w-12 h-12 text-[var(--text-tertiary)] mx-auto mb-4" />
          <h2 className="section-title mb-2">Team comparison</h2>
          <p className="caption">
            Load data for both Men's and Women's teams in the Dashboard to compare.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <p className="caption">Compare Men's and Women's team performance</p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className={`panel p-4 border ${teamStatus?.mens?.loaded ? 'border-[var(--accent-performance)]/40 bg-[var(--accent-performance-muted)]' : 'border-[var(--border-default)]'}`}>
          <div className="flex items-center gap-2 mb-2">
            <Users className={`w-5 h-5 ${teamStatus?.mens?.loaded ? 'text-[var(--accent-performance)]' : 'text-[var(--text-tertiary)]'}`} />
            <h3 className="section-title">Men's Team</h3>
            {teamStatus?.mens?.loaded && <span className="text-xs bg-[var(--accent-performance)] text-white px-2 py-0.5 rounded">Loaded</span>}
          </div>
          {teamStatus?.mens?.loaded ? (
            <p className="text-sm text-[var(--text-secondary)]">Players: <span className="font-semibold text-[var(--text-primary)]">{teamStatus.mens.rowCount}</span></p>
          ) : (
            <p className="caption">No data loaded</p>
          )}
        </div>
        <div className={`panel p-4 border ${teamStatus?.womens?.loaded ? 'border-[var(--accent-performance)]/40 bg-[var(--accent-performance-muted)]' : 'border-[var(--border-default)]'}`}>
          <div className="flex items-center gap-2 mb-2">
            <Users className={`w-5 h-5 ${teamStatus?.womens?.loaded ? 'text-[var(--accent-performance)]' : 'text-[var(--text-tertiary)]'}`} />
            <h3 className="section-title">Women's Team</h3>
            {teamStatus?.womens?.loaded && <span className="text-xs bg-[var(--accent-performance)] text-white px-2 py-0.5 rounded">Loaded</span>}
          </div>
          {teamStatus?.womens?.loaded ? (
            <p className="text-sm text-[var(--text-secondary)]">Players: <span className="font-semibold text-[var(--text-primary)]">{teamStatus.womens.rowCount}</span></p>
          ) : (
            <p className="caption">No data loaded</p>
          )}
        </div>
      </div>

      {/* Comparison Metrics */}
      {showCurrentTeamData ? (
        <div className="panel panel--elevated p-6">
          <h2 className="section-title mb-4 flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-[var(--accent-performance)]" />
            Performance comparison
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
                <div key={idx} className="p-4 rounded border border-[var(--border-default)] bg-[var(--bg-surface)]">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <Icon className="w-5 h-5 text-[var(--accent-performance)]" />
                      <h3 className="section-title">{metric.name}</h3>
                    </div>
                  </div>
                  <div className={`grid gap-4 ${hasBothTeams ? 'grid-cols-2' : 'grid-cols-1'}`}>
                    {(currentTeam === 'mens' && teamStatus?.mens?.loaded && activeKPIs) && (
                      <div className="text-center p-3 rounded border border-[var(--accent-performance)]/20 bg-[var(--accent-performance-muted)]">
                        <p className="caption mb-1">Men's</p>
                        <p className="metric-value text-xl text-[var(--accent-performance)]">
                          {typeof metric.mensValue === 'number' ? metric.mensValue.toFixed(1) : metric.mensValue}
                          {metric.unit && <span className="text-sm ml-1 font-normal">{metric.unit}</span>}
                        </p>
                      </div>
                    )}
                    {(currentTeam === 'womens' && teamStatus?.womens?.loaded && activeKPIs) && (
                      <div className="text-center p-3 rounded border border-[var(--accent-performance)]/20 bg-[var(--accent-performance-muted)]">
                        <p className="caption mb-1">Women's</p>
                        <p className="metric-value text-xl text-[var(--accent-performance)]">
                          {typeof metric.womensValue === 'number' ? metric.womensValue.toFixed(1) : metric.womensValue}
                          {metric.unit && <span className="text-sm ml-1 font-normal">{metric.unit}</span>}
                        </p>
                      </div>
                    )}
                  </div>
                  {hasBothTeams && (
                    <div className="mt-3 pt-3 border-t border-[var(--border-subtle)]">
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-[var(--text-secondary)]">Difference</span>
                        <span className={`font-semibold ${difference >= 0 ? 'text-[var(--accent-performance)]' : 'text-[var(--accent-risk-high)]'}`}>
                          {difference >= 0 ? '+' : ''}{typeof difference === 'number' ? difference.toFixed(1) : difference}
                          {metric.unit && <span className="ml-1">{metric.unit}</span>}
                          <span className="ml-2 text-[var(--text-tertiary)]">({percentageDiff}%)</span>
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      ) : null}

      {/* Additional Insights */}
      {showCurrentTeamData && currentPlayers && (
        <div className="grid grid-cols-1 gap-4">
          {currentTeam === 'mens' && mensPlayers && (
            <div className="panel panel--elevated p-6">
              <h3 className="section-title mb-4">Top — Men's</h3>
              <div className="space-y-2">
                {mensPlayers.slice(0, 5).map((player, idx) => (
                  <div key={player.id} className="flex items-center justify-between py-2 px-2 rounded border border-[var(--border-subtle)] bg-[var(--bg-surface)]">
                    <div className="flex items-center gap-2">
                      <span className="caption">#{idx + 1}</span>
                      <span className="text-sm text-[var(--text-primary)]">{player.name}</span>
                    </div>
                    <span className="font-semibold text-[var(--accent-performance)]">{player.avgLoad}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
          {currentTeam === 'womens' && womensPlayers && (
            <div className="panel panel--elevated p-6">
              <h3 className="section-title mb-4">Top — Women's</h3>
              <div className="space-y-2">
                {womensPlayers.slice(0, 5).map((player, idx) => (
                  <div key={player.id} className="flex items-center justify-between py-2 px-2 rounded border border-[var(--border-subtle)] bg-[var(--bg-surface)]">
                    <div className="flex items-center gap-2">
                      <span className="caption">#{idx + 1}</span>
                      <span className="text-sm text-[var(--text-primary)]">{player.name}</span>
                    </div>
                    <span className="font-semibold text-[var(--accent-performance)]">{player.avgLoad}</span>
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
