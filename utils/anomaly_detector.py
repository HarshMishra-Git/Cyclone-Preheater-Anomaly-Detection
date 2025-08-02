import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
import streamlit as st
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class AnomalyDetector:
    """Advanced anomaly detection for cyclone sensor data."""
    
    def __init__(self):
        self.scalers = {}
        self.models = {}
    
    @st.cache_data
    def detect_zscore_anomalies(_self, df: pd.DataFrame, threshold: float = 3.0, columns: List[str] = None) -> pd.DataFrame:
        """
        Detect anomalies using Z-Score method.
        
        Args:
            df: Input DataFrame
            threshold: Z-score threshold for anomaly detection
            columns: Columns to analyze
            
        Returns:
            DataFrame with anomaly flags and scores
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        result_df = df.copy()
        anomaly_flags = []
        
        for col in columns:
            # Calculate Z-scores
            z_scores = np.abs(stats.zscore(df[col]))
            anomalies = z_scores > threshold
            
            result_df[f'{col}_zscore'] = z_scores
            result_df[f'{col}_anomaly_zscore'] = anomalies
            anomaly_flags.append(anomalies)
        
        # Combined anomaly flag (any column is anomalous)
        result_df['anomaly_zscore'] = np.any(anomaly_flags, axis=0)
        result_df['anomaly_score_zscore'] = np.max([result_df[f'{col}_zscore'] for col in columns], axis=0)
        
        return result_df
    
    @st.cache_data
    def detect_iqr_anomalies(_self, df: pd.DataFrame, multiplier: float = 1.5, columns: List[str] = None) -> pd.DataFrame:
        """
        Detect anomalies using IQR method.
        
        Args:
            df: Input DataFrame
            multiplier: IQR multiplier for outlier detection
            columns: Columns to analyze
            
        Returns:
            DataFrame with anomaly flags
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        result_df = df.copy()
        anomaly_flags = []
        
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            anomalies = (df[col] < lower_bound) | (df[col] > upper_bound)
            
            result_df[f'{col}_anomaly_iqr'] = anomalies
            result_df[f'{col}_iqr_lower'] = lower_bound
            result_df[f'{col}_iqr_upper'] = upper_bound
            anomaly_flags.append(anomalies)
        
        # Combined anomaly flag
        result_df['anomaly_iqr'] = np.any(anomaly_flags, axis=0)
        
        return result_df
    
    @st.cache_resource
    def train_isolation_forest(_self, df: pd.DataFrame, contamination: float = 0.1, columns: List[str] = None) -> IsolationForest:
        """
        Train Isolation Forest model.
        
        Args:
            df: Training DataFrame
            contamination: Expected proportion of anomalies
            columns: Columns to use for training
            
        Returns:
            Trained Isolation Forest model
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Prepare data
        X = df[columns].fillna(df[columns].mean())
        
        # Scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        model.fit(X_scaled)
        
        # Store scaler for later use
        _self.scalers['isolation_forest'] = scaler
        
        return model
    
    def detect_isolation_forest_anomalies(self, df: pd.DataFrame, model: IsolationForest = None, 
                                        contamination: float = 0.1, columns: List[str] = None) -> pd.DataFrame:
        """
        Detect anomalies using Isolation Forest.
        
        Args:
            df: Input DataFrame
            model: Pre-trained model (if None, will train new one)
            contamination: Expected proportion of anomalies
            columns: Columns to analyze
            
        Returns:
            DataFrame with anomaly predictions and scores
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        result_df = df.copy()
        
        # Prepare data
        X = df[columns].fillna(df[columns].mean())
        
        if model is None:
            model = self.train_isolation_forest(df, contamination, columns)
        
        # Scale data using stored scaler
        scaler = self.scalers.get('isolation_forest')
        if scaler is None:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers['isolation_forest'] = scaler
        else:
            X_scaled = scaler.transform(X)
        
        # Predict anomalies
        anomaly_labels = model.predict(X_scaled)
        anomaly_scores = model.decision_function(X_scaled)
        
        # Convert to boolean (True for anomaly)
        result_df['anomaly_isolation_forest'] = anomaly_labels == -1
        result_df['anomaly_score_isolation_forest'] = -anomaly_scores  # Higher score = more anomalous
        
        return result_df
    
    @st.cache_data
    def detect_statistical_anomalies(_self, df: pd.DataFrame, window_size: int = 24, 
                                   threshold: float = 2.0, columns: List[str] = None) -> pd.DataFrame:
        """
        Detect anomalies using rolling statistics.
        
        Args:
            df: Input DataFrame
            window_size: Rolling window size
            threshold: Standard deviation threshold
            columns: Columns to analyze
            
        Returns:
            DataFrame with statistical anomaly flags
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        result_df = df.copy()
        anomaly_flags = []
        
        for col in columns:
            # Calculate rolling statistics
            rolling_mean = df[col].rolling(window=window_size, center=True).mean()
            rolling_std = df[col].rolling(window=window_size, center=True).std()
            
            # Calculate deviations
            deviations = np.abs(df[col] - rolling_mean) / rolling_std
            anomalies = deviations > threshold
            
            result_df[f'{col}_rolling_deviation'] = deviations
            result_df[f'{col}_anomaly_statistical'] = anomalies
            anomaly_flags.append(anomalies.fillna(False))
        
        # Combined anomaly flag
        result_df['anomaly_statistical'] = np.any(anomaly_flags, axis=0)
        
        return result_df
    
    @st.cache_data
    def detect_correlation_anomalies(_self, df: pd.DataFrame, threshold: float = 0.1, 
                                   columns: List[str] = None) -> pd.DataFrame:
        """
        Detect anomalies based on correlation patterns.
        
        Args:
            df: Input DataFrame
            threshold: Correlation threshold for anomaly detection
            columns: Columns to analyze
            
        Returns:
            DataFrame with correlation-based anomaly flags
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        result_df = df.copy()
        
        # Calculate correlation matrix
        corr_matrix = df[columns].corr()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.7:  # High correlation threshold
                    high_corr_pairs.append((columns[i], columns[j], corr_matrix.iloc[i, j]))
        
        # Detect anomalies based on correlation breakdown
        anomaly_flags = []
        
        for col1, col2, expected_corr in high_corr_pairs:
            # Rolling correlation
            rolling_corr = df[col1].rolling(48).corr(df[col2])  # 48 periods rolling
            
            # Detect when correlation breaks down
            corr_anomalies = np.abs(rolling_corr - expected_corr) > threshold
            result_df[f'{col1}_{col2}_corr_anomaly'] = corr_anomalies
            anomaly_flags.append(corr_anomalies.fillna(False))
        
        if anomaly_flags:
            result_df['anomaly_correlation'] = np.any(anomaly_flags, axis=0)
        else:
            result_df['anomaly_correlation'] = False
        
        return result_df
    
    def detect_multivariate_anomalies(self, df: pd.DataFrame, method: str = 'pca', 
                                    contamination: float = 0.1, columns: List[str] = None) -> pd.DataFrame:
        """
        Detect multivariate anomalies using dimensionality reduction.
        
        Args:
            df: Input DataFrame
            method: Method to use ('pca', 'dbscan')
            contamination: Expected proportion of anomalies
            columns: Columns to analyze
            
        Returns:
            DataFrame with multivariate anomaly flags
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        result_df = df.copy()
        X = df[columns].fillna(df[columns].mean())
        
        # Scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if method == 'pca':
            # PCA-based anomaly detection
            pca = PCA(n_components=min(5, len(columns)))
            X_pca = pca.fit_transform(X_scaled)
            
            # Reconstruction error
            X_reconstructed = pca.inverse_transform(X_pca)
            reconstruction_errors = np.sum((X_scaled - X_reconstructed) ** 2, axis=1)
            
            # Threshold based on percentile
            threshold = np.percentile(reconstruction_errors, (1 - contamination) * 100)
            anomalies = reconstruction_errors > threshold
            
            result_df['anomaly_pca'] = anomalies
            result_df['anomaly_score_pca'] = reconstruction_errors
            
        elif method == 'dbscan':
            # DBSCAN clustering
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            cluster_labels = dbscan.fit_predict(X_scaled)
            
            # Points labeled as -1 are anomalies
            anomalies = cluster_labels == -1
            
            result_df['anomaly_dbscan'] = anomalies
            result_df['cluster_label'] = cluster_labels
        
        return result_df
    
    def combine_anomaly_methods(self, df: pd.DataFrame, methods: List[str] = None, 
                              weights: Dict[str, float] = None) -> pd.DataFrame:
        """
        Combine multiple anomaly detection methods.
        
        Args:
            df: DataFrame with anomaly flags from different methods
            methods: List of methods to combine
            weights: Weights for each method
            
        Returns:
            DataFrame with combined anomaly scores
        """
        if methods is None:
            methods = ['zscore', 'iqr', 'isolation_forest', 'statistical']
        
        if weights is None:
            weights = {method: 1.0 for method in methods}
        
        result_df = df.copy()
        
        # Combine binary flags
        combined_flags = []
        for method in methods:
            if f'anomaly_{method}' in df.columns:
                combined_flags.append(df[f'anomaly_{method}'].astype(int) * weights.get(method, 1.0))
        
        if combined_flags:
            result_df['combined_anomaly_score'] = np.sum(combined_flags, axis=0)
            
            # Normalize to 0-1 range
            max_score = sum(weights.values())
            result_df['combined_anomaly_score_normalized'] = result_df['combined_anomaly_score'] / max_score
            
            # Binary flag based on majority vote
            result_df['anomaly_combined'] = result_df['combined_anomaly_score'] >= (max_score / 2)
        
        return result_df
    
    def get_anomaly_summary(self, df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics for detected anomalies.
        
        Args:
            df: DataFrame with anomaly flags
            
        Returns:
            Dictionary with anomaly statistics
        """
        summary = {}
        
        # Find all anomaly columns
        anomaly_columns = [col for col in df.columns if col.startswith('anomaly_')]
        
        for col in anomaly_columns:
            method_name = col.replace('anomaly_', '')
            anomaly_count = df[col].sum() if df[col].dtype == bool else (df[col] == True).sum()
            anomaly_rate = anomaly_count / len(df) * 100
            
            summary[method_name] = {
                'count': int(anomaly_count),
                'rate_percent': round(anomaly_rate, 2),
                'first_occurrence': df[df[col] == True].index.min() if anomaly_count > 0 else None,
                'last_occurrence': df[df[col] == True].index.max() if anomaly_count > 0 else None
            }
        
        return summary
    
    def get_anomaly_periods(self, df: pd.DataFrame, method: str = 'combined', 
                          min_duration_hours: int = 1) -> List[Dict]:
        """
        Extract anomaly periods with start and end times.
        
        Args:
            df: DataFrame with anomaly flags
            method: Anomaly detection method to use
            min_duration_hours: Minimum duration for anomaly periods
            
        Returns:
            List of anomaly periods with metadata
        """
        anomaly_col = f'anomaly_{method}'
        if anomaly_col not in df.columns:
            return []
        
        anomalies = df[anomaly_col].astype(bool)
        periods = []
        
        # Find consecutive anomaly periods
        start_idx = None
        for i, is_anomaly in enumerate(anomalies):
            if is_anomaly and start_idx is None:
                start_idx = i
            elif not is_anomaly and start_idx is not None:
                end_idx = i - 1
                period_duration = df.index[end_idx] - df.index[start_idx]
                
                if period_duration.total_seconds() >= min_duration_hours * 3600:
                    # Calculate severity score safely
                    score_col = f'anomaly_score_{method}'
                    if score_col in df.columns:
                        severity_score = df.iloc[start_idx:end_idx+1][score_col].mean()
                    else:
                        severity_score = 0.0
                    
                    periods.append({
                        'start_time': df.index[start_idx],
                        'end_time': df.index[end_idx],
                        'duration_hours': period_duration.total_seconds() / 3600,
                        'anomaly_count': end_idx - start_idx + 1,
                        'severity_score': severity_score
                    })
                
                start_idx = None
        
        # Handle case where anomaly period extends to end of data
        if start_idx is not None:
            end_idx = len(anomalies) - 1
            period_duration = df.index[end_idx] - df.index[start_idx]
            
            if period_duration.total_seconds() >= min_duration_hours * 3600:
                # Calculate severity score safely
                score_col = f'anomaly_score_{method}'
                if score_col in df.columns:
                    severity_score = df.iloc[start_idx:end_idx+1][score_col].mean()
                else:
                    severity_score = 0.0
                
                periods.append({
                    'start_time': df.index[start_idx],
                    'end_time': df.index[end_idx],
                    'duration_hours': period_duration.total_seconds() / 3600,
                    'anomaly_count': end_idx - start_idx + 1,
                    'severity_score': severity_score
                })
        
        return periods
