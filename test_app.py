import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

class TestGracelandSoccerModel:
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)
        n_samples = 100
        
        data = {
            'Player Name': [f'Player_{i}' for i in range(n_samples)],
            'Player Load': np.random.uniform(100, 500, n_samples),
            'Work Ratio': np.random.uniform(0.5, 1.0, n_samples),
            'Energy (kcal)': np.random.uniform(200, 800, n_samples),
            'Distance (miles)': np.random.uniform(2, 7, n_samples),
            'Sprint Distance (yards)': np.random.uniform(50, 400, n_samples),
            'Top Speed (mph)': np.random.uniform(10, 20, n_samples),
            'Max Acceleration (yd/s/s)': np.random.uniform(2, 8, n_samples),
            'Max Deceleration (yd/s/s)': np.random.uniform(2, 8, n_samples),
            'Distance Per Min (yd/min)': np.random.uniform(50, 150, n_samples),
            'Hr Load': np.random.uniform(100, 500, n_samples),
            'Hr Max (bpm)': np.random.uniform(150, 200, n_samples),
            'Time In Red Zone (min)': np.random.uniform(0, 30, n_samples),
            'Impacts': np.random.randint(0, 50, n_samples),
            'Power Score (w/kg)': np.random.uniform(0.5, 3.0, n_samples),
        }
        
        df = pd.DataFrame(data)
        df['Date'] = pd.date_range(start='2024-01-01', periods=n_samples, freq='D')
        return df
    
    def test_data_loading(self, sample_data):
        """Test data loading functionality"""
        assert not sample_data.empty
        assert len(sample_data) == 100
        assert 'Player Load' in sample_data.columns
    
    def test_outlier_detection(self, sample_data):
        """Test outlier detection"""
        numeric_cols = sample_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col != 'Player Name':
                Q1 = sample_data[col].quantile(0.25)
                Q3 = sample_data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = sample_data[(sample_data[col] < Q1 - 1.5*IQR) | 
                                      (sample_data[col] > Q3 + 1.5*IQR)]
                assert isinstance(outliers, pd.DataFrame)
    
    def test_regression_model_training(self, sample_data):
        """Test regression model can be trained"""
        X = sample_data.select_dtypes(include=[np.number]).drop(columns=['Player Load'], errors='ignore')
        y = sample_data['Player Load']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = GradientBoostingRegressor(n_estimators=10, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        predictions = model.predict(X_test_scaled)
        
        assert len(predictions) == len(y_test)
        assert all(predictions > 0)
    
    def test_missing_values_handling(self, sample_data):
        """Test missing value handling"""
        sample_data.loc[0:5, 'Player Load'] = np.nan
        
        missing_cols = sample_data.columns[sample_data.isnull().any()].tolist()
        assert 'Player Load' in missing_cols or len(missing_cols) > 0
    
    def test_feature_scaling(self, sample_data):
        """Test feature scaling"""
        numeric_cols = sample_data.select_dtypes(include=[np.number]).columns
        data = sample_data[numeric_cols].fillna(0)
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        
        assert scaled_data.mean() < 0.1
        assert scaled_data.std() > 0.9
    
    def test_injury_risk_calculation(self, sample_data):
        """Test injury risk calculation logic"""
        def calculate_injury_risk(df):
            df['Injury_Risk'] = 'Low'
            df.loc[df['Player Load'] > 250, 'Injury_Risk'] = 'Medium'
            df.loc[(df['Player Load'] > 400) & 
                   (df['Hr Load'] > df['Hr Load'].quantile(0.75)), 
                   'Injury_Risk'] = 'High'
            return df
        
        result = calculate_injury_risk(sample_data.copy())
        assert 'Injury_Risk' in result.columns
        assert all(risk in ['Low', 'Medium', 'High'] for risk in result['Injury_Risk'])

if __name__ == '__main__':
    pytest.main([__file__, '-v'])

