import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st
from typing import Optional, Tuple
import logging

class DataProcessor:
    """Handles data loading, cleaning, and preprocessing with performance optimization."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    @st.cache_data
    def load_data(_self, file_path: str, chunk_size: int = 10000) -> pd.DataFrame:
        """
        Load and preprocess cyclone data with caching.
        
        Args:
            file_path: Path to CSV file or uploaded file object
            chunk_size: Size of chunks for processing large files
            
        Returns:
            Processed DataFrame with datetime index
        """
        try:
            # Handle file upload vs file path
            if hasattr(file_path, 'read'):
                # Uploaded file object
                df = pd.read_csv(file_path, low_memory=False)
            else:
                # File path
                df = pd.read_csv(file_path, low_memory=False)
            
            # Process the data
            df = _self._preprocess_data(df)
            
            return df
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            raise e
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the loaded data.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Processed DataFrame
        """
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Handle time column
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
            df = df.set_index('time')
            df = df.sort_index()
        
        # Remove any completely empty rows or columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Convert all columns except time to numeric, handling problematic values
        for col in df.columns:
            if col != 'time':  # Skip time column if present
                # Replace common non-numeric values
                df[col] = df[col].astype(str).replace(['Not Connect', 'not connect', 'N/A', 'NULL', '', 'nan'], np.nan)
                # Convert to numeric, coercing errors to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle missing values with forward fill (appropriate for time series)
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    @st.cache_data
    def get_data_quality_report(_self, df: pd.DataFrame) -> dict:
        """
        Generate data quality report.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with quality metrics
        """
        report = {
            'total_records': len(df),
            'total_variables': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'data_types': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'duplicated_rows': df.duplicated().sum(),
            'date_range': {
                'start': df.index.min() if hasattr(df.index, 'min') else None,
                'end': df.index.max() if hasattr(df.index, 'max') else None,
                'frequency': _self._infer_frequency(df)
            }
        }
        
        # Add statistical summary
        report['statistics'] = df.describe().to_dict()
        
        return report
    
    def _infer_frequency(self, df: pd.DataFrame) -> str:
        """Infer the frequency of time series data."""
        if hasattr(df.index, 'to_series'):
            try:
                freq = pd.infer_freq(df.index)
                return freq if freq else "Irregular"
            except:
                return "Unknown"
        return "Not time series"
    
    @st.cache_data
    def sample_data(_self, df: pd.DataFrame, sample_size: int = 10000, method: str = 'random') -> pd.DataFrame:
        """
        Sample data for faster processing.
        
        Args:
            df: Input DataFrame
            sample_size: Number of samples to return
            method: Sampling method ('random', 'systematic', 'stratified')
            
        Returns:
            Sampled DataFrame
        """
        if len(df) <= sample_size:
            return df
        
        if method == 'random':
            return df.sample(n=sample_size, random_state=42)
        elif method == 'systematic':
            step = len(df) // sample_size
            return df.iloc[::step][:sample_size]
        else:
            return df.sample(n=sample_size, random_state=42)
    
    def detect_outliers_iqr(self, df: pd.DataFrame, columns: list = None, multiplier: float = 1.5) -> pd.DataFrame:
        """
        Detect outliers using IQR method.
        
        Args:
            df: Input DataFrame
            columns: Columns to check for outliers
            multiplier: IQR multiplier for outlier detection
            
        Returns:
            DataFrame with outlier flags
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        outlier_df = df.copy()
        
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            outlier_df[f'{col}_outlier'] = (df[col] < lower_bound) | (df[col] > upper_bound)
        
        return outlier_df
    
    def normalize_data(self, df: pd.DataFrame, method: str = 'minmax') -> pd.DataFrame:
        """
        Normalize numerical data.
        
        Args:
            df: Input DataFrame
            method: Normalization method ('minmax', 'zscore', 'robust')
            
        Returns:
            Normalized DataFrame
        """
        from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df_normalized = df.copy()
        
        if method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'zscore':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError("Method must be 'minmax', 'zscore', or 'robust'")
        
        df_normalized[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        
        return df_normalized
    
    def calculate_rolling_statistics(self, df: pd.DataFrame, window: int = 24) -> pd.DataFrame:
        """
        Calculate rolling statistics for time series data.
        
        Args:
            df: Input DataFrame
            window: Rolling window size
            
        Returns:
            DataFrame with rolling statistics
        """
        result_df = df.copy()
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            result_df[f'{col}_rolling_mean'] = df[col].rolling(window=window).mean()
            result_df[f'{col}_rolling_std'] = df[col].rolling(window=window).std()
            result_df[f'{col}_rolling_min'] = df[col].rolling(window=window).min()
            result_df[f'{col}_rolling_max'] = df[col].rolling(window=window).max()
        
        return result_df
