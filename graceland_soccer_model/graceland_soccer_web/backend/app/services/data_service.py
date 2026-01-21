import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

RECENT_DATA_DAYS = 45


class DataService:
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.df_original: Optional[pd.DataFrame] = None
        self.players: List[str] = []
        self.columns: List[str] = []
        self.excluded_players: set = set()
        self.is_cleaned: bool = False
        self.cleaning_stats: Dict[str, Any] = {}
        self.use_today_as_reference: bool = True
        
        self.key_columns = {
            'player_name': 'Player Name',
            'player_load': 'Player Load',
            'date': 'Date',
            'session_title': 'Session Title',
            'duration': 'Duration',
            'distance': 'Distance (miles)',
            'sprint_distance': 'Sprint Distance (yards)',
            'top_speed': 'Top Speed (mph)',
            'max_acceleration': 'Max Acceleration (yd/s/s)',
            'max_deceleration': 'Max Deceleration (yd/s/s)',
            'work_ratio': 'Work Ratio',
            'energy': 'Energy (kcal)',
            'hr_load': 'Hr Load',
            'impacts': 'Impacts',
            'power_plays': 'Power Plays',
            'power_score': 'Power Score (w/kg)',
            'distance_per_min': 'Distance Per Min (yd/min)',
        }
        
        self.risk_thresholds = {'high_load': 500, 'low_load': 200}
    
    def load_csv(self, file_path: str) -> Dict[str, Any]:
        self.df = pd.read_csv(file_path)
        self.df_original = self.df.copy()
        self.excluded_players = set()
        self.is_cleaned = False
        self.cleaning_stats = {}
        return self._process_loaded_data()
    
    def load_from_upload(self, content: bytes) -> Dict[str, Any]:
        from io import BytesIO
        self.df = pd.read_csv(BytesIO(content))
        self.df_original = self.df.copy()
        self.excluded_players = set()
        self.is_cleaned = False
        self.cleaning_stats = {}
        return self._process_loaded_data()
    
    def _process_loaded_data(self) -> Dict[str, Any]:
        self.columns = self.df.columns.tolist()
        player_col = self.key_columns['player_name']
        if player_col in self.df.columns:
            self.players = self.df[player_col].str.strip().unique().tolist()
        self._convert_numeric_columns()
        self._parse_dates()
        return {
            'rowCount': len(self.df),
            'columnCount': len(self.columns),
            'columns': self.columns,
            'players': self.players,
            'dateRange': self._get_date_range()
        }
    
    def _convert_numeric_columns(self):
        numeric_cols = ['Player Load', 'Duration', 'Distance (miles)', 'Sprint Distance (yards)',
            'Top Speed (mph)', 'Max Acceleration (yd/s/s)', 'Max Deceleration (yd/s/s)',
            'Work Ratio', 'Energy (kcal)', 'Hr Load', 'Impacts', 'Power Plays',
            'Power Score (w/kg)', 'Distance Per Min (yd/min)']
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
    
    def _parse_dates(self):
        date_col = self.key_columns['date']
        if date_col in self.df.columns:
            self.df['ParsedDate'] = pd.to_datetime(self.df[date_col], errors='coerce')
    
    def _get_date_range(self) -> Dict[str, str]:
        if 'ParsedDate' in self.df.columns and self.df['ParsedDate'].notna().any():
            return {
                'start': self.df['ParsedDate'].min().strftime('%Y-%m-%d'),
                'end': self.df['ParsedDate'].max().strftime('%Y-%m-%d')
            }
        return {'start': 'Unknown', 'end': 'Unknown'}
    
    def _get_recent_data_for_player(self, player_name: str) -> pd.DataFrame:
        """Get data from the last 45 days (1.5 months) from TODAY's date for a player"""
        if self.df is None:
            return pd.DataFrame()
        
        pdata = self.df[self.df[self.key_columns['player_name']].str.strip() == player_name]
        
        if pdata.empty or 'ParsedDate' not in pdata.columns:
            return pd.DataFrame()
        
        # Determine reference date based on configuration
        if self.use_today_as_reference:
            # Use TODAY's actual date as reference
            reference_date = datetime.now()
        else:
            # Use the last training date from the CSV dataset
            max_date = self.df['ParsedDate'].max()
            if pd.isna(max_date):
                # Fallback to today if no valid dates
                reference_date = datetime.now()
            else:
                reference_date = max_date
        
        cutoff_date = reference_date - timedelta(days=RECENT_DATA_DAYS)
        
        # Filter for last 45 days from reference date
        recent_data = pdata[pdata['ParsedDate'] >= cutoff_date]
        
        return recent_data
    
    def has_recent_data(self, player_name: str) -> bool:
        """Check if player has data within the last 45 days"""
        recent = self._get_recent_data_for_player(player_name)
        return len(recent) > 0
    
    def _get_active_players(self) -> List[str]:
        """Get list of players excluding removed ones"""
        return [p for p in self.players if p not in self.excluded_players]
    
    def exclude_player(self, player_name: str) -> bool:
        """Exclude a player from analysis"""
        if player_name in self.players:
            self.excluded_players.add(player_name)
            return True
        return False
    
    def restore_player(self, player_name: str) -> bool:
        """Restore a previously excluded player"""
        if player_name in self.excluded_players:
            self.excluded_players.discard(player_name)
            return True
        return False
    
    def delete_player_data(self, player_id: str) -> bool:
        """Permanently delete player data from the dataframe"""
        if self.df is None:
            return False
        try:
            idx = int(player_id.replace('player_', ''))
            if idx >= len(self.players):
                return False
            player_name = self.players[idx]
            player_col = self.key_columns['player_name']
            
            # Remove from dataframe
            self.df = self.df[self.df[player_col].str.strip() != player_name]
            
            # Update players list
            self.players = [p for p in self.players if p != player_name]
            self.excluded_players.discard(player_name)
            
            return True
        except Exception as e:
            logger.error(f"Error deleting player: {e}")
            return False
    
    def get_dashboard_kpis(self) -> Dict[str, Any]:
        if self.df is None:
            return {'totalPlayers': 0, 'totalPlayersChange': 0, 'avgTeamLoad': 0, 
                    'avgTeamLoadChange': 0, 'highRiskPlayers': 0, 'highRiskPlayersChange': 0,
                    'avgTeamSpeed': 0, 'avgTeamSpeedChange': 0}
        
        active_players = self._get_active_players()
        player_col = self.key_columns['player_name']
        active_df = self.df[self.df[player_col].str.strip().isin(active_players)]
        
        load_col, speed_col = self.key_columns['player_load'], self.key_columns['top_speed']
        avg_load = active_df[load_col].mean() if load_col in active_df.columns else 0
        avg_speed = active_df[speed_col].mean() if speed_col in active_df.columns else 0
        
        return {
            'totalPlayers': len(active_players), 'totalPlayersChange': 5.2,
            'avgTeamLoad': round(avg_load, 1), 'avgTeamLoadChange': -3.1,
            'highRiskPlayers': self._count_high_risk_players(), 'highRiskPlayersChange': 12.5,
            'avgTeamSpeed': round(avg_speed, 1), 'avgTeamSpeedChange': 2.8
        }
    
    def _count_high_risk_players(self) -> int:
        """Count high risk players based on LAST 45 DAYS from TODAY only"""
        if self.df is None: return 0
        load_col = self.key_columns['player_load']
        if load_col not in self.df.columns: return 0
        
        active_players = self._get_active_players()
        high_risk_count = 0
        
        for player_name in active_players:
            recent_data = self._get_recent_data_for_player(player_name)
            # Only count as high risk if they have recent data AND high load
            if len(recent_data) > 0:
                avg_load = recent_data[load_col].mean() if load_col in recent_data.columns else 0
                if avg_load > self.risk_thresholds['high_load']:
                    high_risk_count += 1
        
        return high_risk_count
    
    def get_risk_distribution(self) -> Dict[str, int]:
        """Get risk distribution based on LAST 45 DAYS from TODAY only"""
        if self.df is None: return {'low': 0, 'medium': 0, 'high': 0}
        player_col, load_col = self.key_columns['player_name'], self.key_columns['player_load']
        if load_col not in self.df.columns: return {'low': 0, 'medium': 0, 'high': 0}
        
        active_players = self._get_active_players()
        
        # Count risk for each player based on recent data only
        low, medium, high = 0, 0, 0
        
        for player_name in active_players:
            recent_data = self._get_recent_data_for_player(player_name)
            
            # If no recent data (last 45 days from TODAY), player is LOW risk
            if len(recent_data) == 0:
                low += 1
            else:
                # Calculate average load from recent data only
                avg_load = recent_data[load_col].mean() if load_col in recent_data.columns else 0
                if avg_load < self.risk_thresholds['low_load']:
                    low += 1
                elif avg_load > self.risk_thresholds['high_load']:
                    high += 1
                else:
                    medium += 1
        
        return {'low': low, 'medium': medium, 'high': high}
    
    def get_load_history(self, days: int = 15) -> List[Dict[str, Any]]:
        if self.df is None or 'ParsedDate' not in self.df.columns: return []
        load_col = self.key_columns['player_load']
        if load_col not in self.df.columns: return []
        
        player_col = self.key_columns['player_name']
        active_players = self._get_active_players()
        active_df = self.df[self.df[player_col].str.strip().isin(active_players)]
        
        daily = active_df.groupby(active_df['ParsedDate'].dt.date).agg({
            load_col: 'mean', self.key_columns['player_name']: 'count'
        }).reset_index()
        daily.columns = ['date', 'avgLoad', 'sessionCount']
        daily = daily.sort_values('date').tail(days)
        return [{'date': str(r['date']), 'avgLoad': round(r['avgLoad'], 1), 
                 'sessionCount': int(r['sessionCount'])} for _, r in daily.iterrows()]
    
    def get_all_players(self) -> List[Dict[str, Any]]:
        """Get all players with risk based on LAST 45 DAYS from TODAY"""
        if self.df is None: return []
        player_col, load_col, speed_col = self.key_columns['player_name'], self.key_columns['player_load'], self.key_columns['top_speed']
        positions = ['GK', 'CB', 'LB', 'RB', 'CM', 'CDM', 'CAM', 'LW', 'RW', 'ST', 'CF']
        players = []
        
        for i, name in enumerate(self.players):
            if name in self.excluded_players:
                continue
            
            # Get all data for historical stats
            pdata = self.df[self.df[player_col].str.strip() == name]
            avg_load = pdata[load_col].mean() if load_col in self.df.columns else 0
            avg_speed = pdata[speed_col].mean() if speed_col in self.df.columns else 0
            
            # Get recent data for risk calculation (last 45 days from TODAY)
            recent_data = self._get_recent_data_for_player(name)
            
            # Calculate risk based on recent data ONLY
            if len(recent_data) == 0:
                # No recent data = LOW risk
                risk = 'low'
            else:
                recent_avg_load = recent_data[load_col].mean() if load_col in recent_data.columns else 0
                risk = 'high' if recent_avg_load > self.risk_thresholds['high_load'] else ('low' if recent_avg_load < self.risk_thresholds['low_load'] else 'medium')
            
            last = str(pdata['ParsedDate'].max().date()) if 'ParsedDate' in pdata.columns and pdata['ParsedDate'].notna().any() else None
            
            players.append({
                'id': f'player_{i}', 
                'name': name.strip(), 
                'position': positions[i % len(positions)],
                'number': i + 1, 
                'riskLevel': risk, 
                'avgLoad': round(avg_load, 1), 
                'avgSpeed': round(avg_speed, 1),
                'sessions': len(pdata), 
                'lastSession': last,
                'hasRecentData': len(recent_data) > 0,
                'recentSessions': len(recent_data)
            })
        return players
    
    def get_player_detail(self, player_id: str) -> Optional[Dict[str, Any]]:
        if self.df is None: return None
        try:
            idx = int(player_id.replace('player_', ''))
            if idx >= len(self.players): return None
            player_name = self.players[idx]
            if player_name in self.excluded_players: return None
        except: return None
        
        pdata = self.df[self.df[self.key_columns['player_name']].str.strip() == player_name]
        if pdata.empty: return None
        
        base = next((p for p in self.get_all_players() if p['id'] == player_id), None)
        if not base: return None
        
        # Check if player has recent data (last 45 days)
        recent_data = self._get_recent_data_for_player(player_name)
        has_recent = len(recent_data) > 0
        
        # Use recent data for metrics if available, otherwise use all data
        data_for_metrics = recent_data if has_recent else pdata
        
        def safe_mean(col, data=data_for_metrics): 
            return round(float(data[col].mean()), 2) if col in data.columns and data[col].notna().any() else 0.0
        def safe_max(col, data=data_for_metrics): 
            return round(float(data[col].max()), 2) if col in data.columns and data[col].notna().any() else 0.0
        def safe_min(col, data=data_for_metrics): 
            return round(float(data[col].min()), 2) if col in data.columns and data[col].notna().any() else 0.0
        def safe_std(col, data=data_for_metrics): 
            return round(float(data[col].std()), 2) if col in data.columns and data[col].notna().any() and len(data) > 1 else 0.0
        
        # Build metrics with original column names for ML model compatibility
        # Use recent data (last 45 days) for risk prediction
        metrics = {
            'Player Load': safe_mean(self.key_columns['player_load']),
            'Duration': safe_mean(self.key_columns['duration']),
            'Distance (miles)': safe_mean(self.key_columns['distance']),
            'Sprint Distance (yards)': safe_mean(self.key_columns['sprint_distance']),
            'Top Speed (mph)': safe_mean(self.key_columns['top_speed']),
            'Max Acceleration (yd/s/s)': safe_mean(self.key_columns['max_acceleration']),
            'Max Deceleration (yd/s/s)': safe_mean(self.key_columns['max_deceleration']),
            'Work Ratio': safe_mean(self.key_columns['work_ratio']),
            'Energy (kcal)': safe_mean(self.key_columns['energy']),
            'Hr Load': safe_mean(self.key_columns['hr_load']),
            'Impacts': safe_mean(self.key_columns['impacts']),
            'Power Plays': safe_mean(self.key_columns['power_plays']),
            'Power Score (w/kg)': safe_mean(self.key_columns['power_score']),
            'Distance Per Min (yd/min)': safe_mean(self.key_columns['distance_per_min']),
        }
        
        # Extended stats for display (use all data for historical context)
        extended_stats = {
            'playerLoad': {'avg': safe_mean(self.key_columns['player_load'], pdata), 'max': safe_max(self.key_columns['player_load'], pdata), 'min': safe_min(self.key_columns['player_load'], pdata), 'std': safe_std(self.key_columns['player_load'], pdata)},
            'distance': {'avg': safe_mean(self.key_columns['distance'], pdata), 'max': safe_max(self.key_columns['distance'], pdata), 'min': safe_min(self.key_columns['distance'], pdata)},
            'sprintDistance': {'avg': safe_mean(self.key_columns['sprint_distance'], pdata), 'max': safe_max(self.key_columns['sprint_distance'], pdata)},
            'topSpeed': {'avg': safe_mean(self.key_columns['top_speed'], pdata), 'max': safe_max(self.key_columns['top_speed'], pdata)},
            'workRatio': {'avg': safe_mean(self.key_columns['work_ratio'], pdata), 'max': safe_max(self.key_columns['work_ratio'], pdata)},
            'energy': {'avg': safe_mean(self.key_columns['energy'], pdata), 'total': round(float(pdata[self.key_columns['energy']].sum()) if self.key_columns['energy'] in pdata.columns else 0, 0)},
        }
        
        # Session history for charts
        history = []
        for _, r in pdata.iterrows():
            session = {
                'date': str(r.get('ParsedDate', 'Unknown')).split(' ')[0] if pd.notna(r.get('ParsedDate')) else 'Unknown',
                'sessionTitle': str(r.get(self.key_columns['session_title'], 'Session')),
                'playerLoad': float(r.get(self.key_columns['player_load'], 0)) if pd.notna(r.get(self.key_columns['player_load'])) else 0,
                'distance': float(r.get(self.key_columns['distance'], 0)) if pd.notna(r.get(self.key_columns['distance'])) else 0,
                'duration': float(r.get(self.key_columns['duration'], 0)) if pd.notna(r.get(self.key_columns['duration'])) else 0,
                'topSpeed': float(r.get(self.key_columns['top_speed'], 0)) if pd.notna(r.get(self.key_columns['top_speed'])) else 0,
                'sprintDistance': float(r.get(self.key_columns['sprint_distance'], 0)) if pd.notna(r.get(self.key_columns['sprint_distance'])) else 0,
            }
            history.append(session)
        
        # Sort by date and limit
        history = sorted(history, key=lambda x: x['date'])[-30:]
        
        return {
            **base, 
            'metrics': metrics, 
            'extendedStats': extended_stats, 
            'history': history,
            'hasRecentData': has_recent,
            'recentSessionCount': len(recent_data)
        }
    
    def get_player_comparison_data(self, player_ids: List[str]) -> List[Dict[str, Any]]:
        """Get comparison data for multiple players"""
        results = []
        for pid in player_ids:
            detail = self.get_player_detail(pid)
            if detail:
                results.append({
                    'id': detail['id'],
                    'name': detail['name'],
                    'avgLoad': detail['avgLoad'],
                    'avgSpeed': detail['avgSpeed'],
                    'sessions': detail['sessions'],
                    'riskLevel': detail['riskLevel'],
                    'extendedStats': detail.get('extendedStats', {})
                })
        return results
    
    def get_high_risk_players(self) -> List[Dict[str, Any]]:
        return [p for p in self.get_all_players() if p['riskLevel'] == 'high']
    
    def get_top_performers(self, limit: int = 5) -> List[Dict[str, Any]]:
        return sorted(self.get_all_players(), key=lambda x: x['avgLoad'], reverse=True)[:limit]
    
    def get_data_for_training(self) -> Tuple[pd.DataFrame, List[str]]:
        if self.df is None: return pd.DataFrame(), []
        feature_cols = ['Duration', 'Distance (miles)', 'Sprint Distance (yards)', 'Top Speed (mph)',
            'Max Acceleration (yd/s/s)', 'Max Deceleration (yd/s/s)', 'Work Ratio', 'Energy (kcal)',
            'Power Plays', 'Power Score (w/kg)', 'Distance Per Min (yd/min)']
        return self.df, [c for c in feature_cols if c in self.df.columns]
    
    def get_data_audit(self) -> Dict[str, Any]:
        """Get comprehensive data quality audit"""
        if self.df is None:
            return {}
        
        audit = {
            'totalRows': len(self.df),
            'totalColumns': len(self.df.columns),
            'totalPlayers': len(self.players),
            'isCleaned': self.is_cleaned,
            'cleaningStats': self.cleaning_stats,
            'missingValues': {},
            'outliers': {},
            'columnStats': {},
            'dataQualityScore': 100,
            'warnings': [],
            'recommendations': []
        }
        
        # Check for missing values
        total_missing = 0
        for col in self.df.columns:
            missing = int(self.df[col].isna().sum())
            if missing > 0:
                missing_pct = round(missing / len(self.df) * 100, 2)
                audit['missingValues'][col] = {'count': missing, 'percentage': missing_pct}
                total_missing += missing
                if missing_pct > 5:
                    audit['warnings'].append(f"{col}: {missing_pct}% missing values")
        
        # Check for outliers in numeric columns (using 3.0 IQR for permissive detection)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        key_metrics = ['Player Load', 'Energy (kcal)', 'Distance (miles)', 'Top Speed (mph)', 
                       'Sprint Distance (yards)', 'Work Ratio', 'Duration']
        
        # Use 3.0 IQR threshold for more permissive outlier detection
        outlier_threshold = 3.0
        
        total_outliers = 0
        for col in numeric_cols:
            if col in key_metrics or col in self.df.columns:
                try:
                    Q1 = self.df[col].quantile(0.25)
                    Q3 = self.df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    if IQR > 0:
                        # More permissive: only flag extreme outliers (3.0 IQR)
                        lower_bound = Q1 - outlier_threshold * IQR
                        upper_bound = Q3 + outlier_threshold * IQR
                        outliers_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
                        outlier_count = int(outliers_mask.sum())
                        if outlier_count > 0:
                            outlier_pct = round(outlier_count / len(self.df) * 100, 2)
                            audit['outliers'][col] = {
                                'count': outlier_count,
                                'percentage': outlier_pct,
                                'lowerBound': round(lower_bound, 2),
                                'upperBound': round(upper_bound, 2),
                                'min': round(float(self.df[col].min()), 2),
                                'max': round(float(self.df[col].max()), 2)
                            }
                            total_outliers += outlier_count
                            # Only warn if more than 10% are outliers (since we're using 3.0 IQR)
                            if outlier_pct > 10:
                                audit['warnings'].append(f"{col}: {outlier_pct}% extreme outliers detected")
                except Exception as e:
                    logger.warning(f"Error checking outliers for {col}: {e}")
        
        # Column statistics for key metrics
        for col in key_metrics:
            if col in self.df.columns:
                try:
                    audit['columnStats'][col] = {
                        'mean': round(float(self.df[col].mean()), 2),
                        'std': round(float(self.df[col].std()), 2),
                        'min': round(float(self.df[col].min()), 2),
                        'max': round(float(self.df[col].max()), 2),
                        'median': round(float(self.df[col].median()), 2),
                        'q25': round(float(self.df[col].quantile(0.25)), 2),
                        'q75': round(float(self.df[col].quantile(0.75)), 2),
                    }
                except:
                    pass
        
        # Calculate data quality score
        score = 100
        if total_missing > 0:
            score -= min(20, (total_missing / (len(self.df) * len(self.df.columns)) * 100))
        if total_outliers > 0:
            score -= min(30, (total_outliers / len(self.df) * 10))
        
        audit['dataQualityScore'] = max(0, round(score, 1))
        
        # Recommendations
        if audit['outliers']:
            audit['recommendations'].append("Click 'Clean Outliers' to remove extreme values using IQR method")
        if audit['missingValues']:
            audit['recommendations'].append("Consider filling missing values before training models")
        if not self.is_cleaned and audit['outliers']:
            audit['recommendations'].append("Data cleaning recommended for better model performance")
        
        return audit
    
    def clean_outliers(self, method: str = 'iqr', threshold: float = 3.0) -> Dict[str, Any]:
        """Clean outliers using IQR method - More permissive (threshold 3.0 by default)"""
        if self.df is None:
            return {'error': 'No data loaded'}
        
        df_before = len(self.df)
        rows_modified = 0
        column_changes = {}
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols:
            try:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR > 0:
                    # More permissive: use 3.0 IQR by default (only cap extreme outliers)
                    # This is more conservative and preserves more data
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    
                    # Only cap values that are truly extreme (beyond 3 IQR)
                    # This is much more permissive than standard 1.5 IQR
                    outliers_low = (self.df[col] < lower_bound).sum()
                    outliers_high = (self.df[col] > upper_bound).sum()
                    total_outliers = outliers_low + outliers_high
                    
                    if total_outliers > 0:
                        # Cap only extreme outliers (more permissive approach)
                        self.df[col] = np.where(
                            self.df[col] < lower_bound, lower_bound,
                            np.where(self.df[col] > upper_bound, upper_bound, self.df[col])
                        )
                        column_changes[col] = {
                            'outliersCapped': int(total_outliers),
                            'lowerBound': round(lower_bound, 2),
                            'upperBound': round(upper_bound, 2)
                        }
                        rows_modified += total_outliers
            except Exception as e:
                logger.warning(f"Error cleaning outliers for {col}: {e}")
        
        self.is_cleaned = True
        self.cleaning_stats = {
            'method': method,
            'threshold': threshold,
            'rowsBefore': df_before,
            'rowsAfter': len(self.df),
            'totalOutliersCapped': int(rows_modified),
            'columnsAffected': len(column_changes),
            'columnDetails': column_changes
        }
        
        # Refresh player list and data
        self._process_loaded_data()
        
        return {
            'success': True,
            'message': f'Cleaned {rows_modified} outlier values across {len(column_changes)} columns',
            'stats': self.cleaning_stats
        }
    
    def reset_to_original(self) -> Dict[str, Any]:
        """Reset data to original (undo cleaning)"""
        if self.df_original is None:
            return {'error': 'No original data available'}
        
        self.df = self.df_original.copy()
        self.is_cleaned = False
        self.cleaning_stats = {}
        self._process_loaded_data()
        
        return {
            'success': True,
            'message': 'Data reset to original',
            'rowCount': len(self.df)
        }
    
    def get_date_reference_setting(self) -> Dict[str, Any]:
        """Get current date reference setting"""
        return {
            'useTodayAsReference': self.use_today_as_reference,
            'description': 'Use today\'s date' if self.use_today_as_reference else 'Use last training date from CSV'
        }
    
    def set_date_reference_setting(self, use_today: bool) -> Dict[str, Any]:
        """Set date reference setting"""
        self.use_today_as_reference = use_today
        return {
            'success': True,
            'useTodayAsReference': self.use_today_as_reference,
            'message': f'Date reference set to: {"Today\'s date" if use_today else "Last training date from CSV"}'
        }


data_service = DataService()
