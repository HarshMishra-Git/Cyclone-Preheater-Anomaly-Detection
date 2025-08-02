import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class Visualizer:
    """Advanced visualization utilities for cyclone data analysis."""
    
    def __init__(self):
        self.color_palette = {
            'normal': '#1f77b4',
            'anomaly': '#d62728',
            'warning': '#ff7f0e',
            'good': '#2ca02c',
            'background': '#f8f9fa'
        }
    
    def create_time_series_plot(self, df: pd.DataFrame, columns: List[str] = None, 
                              title: str = None, height: int = 500) -> go.Figure:
        """
        Create interactive time series plot.
        
        Args:
            df: Input DataFrame with datetime index
            columns: Columns to plot
            title: Plot title
            height: Plot height
            
        Returns:
            Plotly figure
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()[:6]  # Limit to 6 for readability
        elif isinstance(columns, str):
            columns = [columns]
        
        fig = go.Figure()
        
        for col in columns:
            if col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df[col],
                    mode='lines',
                    name=col,
                    line=dict(width=1.5),
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                'Time: %{x}<br>' +
                                'Value: %{y:.2f}<br>' +
                                '<extra></extra>'
                ))
        
        fig.update_layout(
            title=title or f"Time Series: {', '.join(columns)}",
            xaxis_title="Time",
            yaxis_title="Value",
            height=height,
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def create_anomaly_plot(self, df: pd.DataFrame, column: str, anomaly_column: str = None,
                          title: str = None, height: int = 500) -> go.Figure:
        """
        Create time series plot with anomalies highlighted.
        
        Args:
            df: Input DataFrame
            column: Data column to plot
            anomaly_column: Column with anomaly flags
            title: Plot title
            height: Plot height
            
        Returns:
            Plotly figure with highlighted anomalies
        """
        if anomaly_column is None:
            anomaly_column = f'anomaly_{column}' if f'anomaly_{column}' in df.columns else 'anomaly_combined'
        
        fig = go.Figure()
        
        # Normal data points
        normal_mask = ~df[anomaly_column] if anomaly_column in df.columns else [True] * len(df)
        fig.add_trace(go.Scatter(
            x=df[normal_mask].index,
            y=df[normal_mask][column],
            mode='lines',
            name='Normal',
            line=dict(color=self.color_palette['normal'], width=1.5),
            hovertemplate='<b>Normal</b><br>' +
                        'Time: %{x}<br>' +
                        'Value: %{y:.2f}<br>' +
                        '<extra></extra>'
        ))
        
        # Anomalous data points
        if anomaly_column in df.columns:
            anomaly_mask = df[anomaly_column]
            if anomaly_mask.any():
                fig.add_trace(go.Scatter(
                    x=df[anomaly_mask].index,
                    y=df[anomaly_mask][column],
                    mode='markers',
                    name='Anomaly',
                    marker=dict(
                        color=self.color_palette['anomaly'],
                        size=8,
                        symbol='diamond'
                    ),
                    hovertemplate='<b>Anomaly</b><br>' +
                                'Time: %{x}<br>' +
                                'Value: %{y:.2f}<br>' +
                                '<extra></extra>'
                ))
        
        fig.update_layout(
            title=title or f"{column} - Anomaly Detection",
            xaxis_title="Time",
            yaxis_title=column,
            height=height,
            hovermode='x unified',
            showlegend=True
        )
        
        return fig
    
    def create_correlation_heatmap(self, df: pd.DataFrame, columns: List[str] = None,
                                 title: str = "Correlation Matrix") -> go.Figure:
        """
        Create correlation heatmap.
        
        Args:
            df: Input DataFrame
            columns: Columns to include in correlation
            title: Plot title
            
        Returns:
            Plotly heatmap figure
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        corr_matrix = df[columns].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hovertemplate='<b>%{y} vs %{x}</b><br>' +
                        'Correlation: %{z:.3f}<br>' +
                        '<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Variables",
            yaxis_title="Variables",
            height=max(400, len(columns) * 40)
        )
        
        return fig
    
    def create_distribution_plot(self, df: pd.DataFrame, column: str, 
                               bins: int = 50, title: str = None) -> go.Figure:
        """
        Create distribution plot with statistics.
        
        Args:
            df: Input DataFrame
            column: Column to plot
            bins: Number of histogram bins
            title: Plot title
            
        Returns:
            Plotly figure with histogram and statistics
        """
        fig = go.Figure()
        
        # Histogram
        fig.add_trace(go.Histogram(
            x=df[column],
            nbinsx=bins,
            name='Distribution',
            opacity=0.7,
            marker_color=self.color_palette['normal']
        ))
        
        # Add statistics lines
        mean_val = df[column].mean()
        median_val = df[column].median()
        std_val = df[column].std()
        
        fig.add_vline(x=mean_val, line_dash="dash", line_color="red", 
                     annotation_text=f"Mean: {mean_val:.2f}")
        fig.add_vline(x=median_val, line_dash="dot", line_color="green",
                     annotation_text=f"Median: {median_val:.2f}")
        fig.add_vline(x=mean_val + 2*std_val, line_dash="dash", line_color="orange",
                     annotation_text=f"+2σ: {mean_val + 2*std_val:.2f}")
        fig.add_vline(x=mean_val - 2*std_val, line_dash="dash", line_color="orange",
                     annotation_text=f"-2σ: {mean_val - 2*std_val:.2f}")
        
        fig.update_layout(
            title=title or f"Distribution: {column}",
            xaxis_title=column,
            yaxis_title="Frequency",
            showlegend=False
        )
        
        return fig
    
    def create_anomaly_score_plot(self, df: pd.DataFrame, score_columns: List[str],
                                title: str = "Anomaly Scores") -> go.Figure:
        """
        Create anomaly score comparison plot.
        
        Args:
            df: Input DataFrame
            score_columns: Columns with anomaly scores
            title: Plot title
            
        Returns:
            Plotly figure with anomaly scores
        """
        fig = make_subplots(
            rows=len(score_columns), cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=score_columns
        )
        
        colors = px.colors.qualitative.Set1
        
        for i, col in enumerate(score_columns):
            if col in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[col],
                        mode='lines',
                        name=col,
                        line=dict(color=colors[i % len(colors)]),
                        showlegend=False
                    ),
                    row=i+1, col=1
                )
        
        fig.update_layout(
            title=title,
            height=150 * len(score_columns),
            hovermode='x unified'
        )
        
        return fig
    
    def create_anomaly_summary_chart(self, anomaly_summary: Dict) -> go.Figure:
        """
        Create bar chart summarizing anomaly detection results.
        
        Args:
            anomaly_summary: Dictionary with anomaly statistics
            
        Returns:
            Plotly bar chart
        """
        methods = list(anomaly_summary.keys())
        counts = [anomaly_summary[method]['count'] for method in methods]
        rates = [anomaly_summary[method]['rate_percent'] for method in methods]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Anomaly Counts', 'Anomaly Rates (%)'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Anomaly counts
        fig.add_trace(
            go.Bar(
                x=methods,
                y=counts,
                name='Count',
                marker_color=self.color_palette['anomaly'],
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Anomaly rates
        fig.add_trace(
            go.Bar(
                x=methods,
                y=rates,
                name='Rate (%)',
                marker_color=self.color_palette['warning'],
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Anomaly Detection Summary",
            height=400
        )
        
        return fig
    
    def create_multivariate_plot(self, df: pd.DataFrame, columns: List[str],
                               anomaly_column: str = 'anomaly_combined',
                               title: str = "Multivariate Analysis") -> go.Figure:
        """
        Create 3D scatter plot for multivariate anomaly visualization.
        
        Args:
            df: Input DataFrame
            columns: Columns for 3D plot (max 3)
            anomaly_column: Column with anomaly flags
            title: Plot title
            
        Returns:
            Plotly 3D scatter plot
        """
        columns = columns[:3]  # Limit to 3 dimensions
        
        if len(columns) < 3:
            # Use PCA or repeat columns if needed
            while len(columns) < 3:
                columns.append(columns[0])
        
        normal_mask = ~df[anomaly_column] if anomaly_column in df.columns else [True] * len(df)
        anomaly_mask = df[anomaly_column] if anomaly_column in df.columns else [False] * len(df)
        
        fig = go.Figure()
        
        # Normal points
        fig.add_trace(go.Scatter3d(
            x=df[normal_mask][columns[0]],
            y=df[normal_mask][columns[1]],
            z=df[normal_mask][columns[2]],
            mode='markers',
            name='Normal',
            marker=dict(
                size=3,
                color=self.color_palette['normal'],
                opacity=0.6
            )
        ))
        
        # Anomalous points
        if anomaly_mask.any():
            fig.add_trace(go.Scatter3d(
                x=df[anomaly_mask][columns[0]],
                y=df[anomaly_mask][columns[1]],
                z=df[anomaly_mask][columns[2]],
                mode='markers',
                name='Anomaly',
                marker=dict(
                    size=6,
                    color=self.color_palette['anomaly'],
                    symbol='diamond',
                    opacity=0.8
                )
            ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=columns[0],
                yaxis_title=columns[1],
                zaxis_title=columns[2]
            ),
            height=600
        )
        
        return fig
    
    def create_anomaly_timeline(self, anomaly_periods: List[Dict], 
                              title: str = "Anomaly Timeline") -> go.Figure:
        """
        Create timeline visualization of anomaly periods.
        
        Args:
            anomaly_periods: List of anomaly period dictionaries
            title: Plot title
            
        Returns:
            Plotly timeline figure
        """
        if not anomaly_periods:
            fig = go.Figure()
            fig.add_annotation(
                text="No anomaly periods detected",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=16
            )
            fig.update_layout(title=title, height=300)
            return fig
        
        fig = go.Figure()
        
        for i, period in enumerate(anomaly_periods):
            # Calculate color based on severity
            severity = period.get('severity_score', 0)
            color_intensity = min(1.0, max(0.3, severity / 10))  # Normalize to 0.3-1.0
            
            fig.add_trace(go.Scatter(
                x=[period['start_time'], period['end_time']],
                y=[i, i],
                mode='lines+markers',
                name=f"Period {i+1}",
                line=dict(width=8, color=f'rgba(214, 39, 40, {color_intensity})'),
                marker=dict(size=10),
                hovertemplate=f'<b>Anomaly Period {i+1}</b><br>' +
                            f'Start: {period["start_time"]}<br>' +
                            f'End: {period["end_time"]}<br>' +
                            f'Duration: {period["duration_hours"]:.1f} hours<br>' +
                            f'Count: {period["anomaly_count"]}<br>' +
                            f'Severity: {period.get("severity_score", 0):.2f}<br>' +
                            '<extra></extra>'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Anomaly Periods",
            yaxis=dict(tickmode='linear', tick0=0, dtick=1),
            height=max(300, len(anomaly_periods) * 50),
            showlegend=False
        )
        
        return fig
    
    def create_system_overview_dashboard(self, df: pd.DataFrame, 
                                       anomaly_summary: Dict) -> go.Figure:
        """
        Create comprehensive system overview dashboard.
        
        Args:
            df: Input DataFrame
            anomaly_summary: Anomaly detection summary
            
        Returns:
            Plotly dashboard figure
        """
        # Get main sensor columns
        sensor_cols = [col for col in df.columns if 'Temp' in col or 'draft' in col][:4]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('System Temperature Overview', 'Pressure Draft Overview',
                          'Anomaly Detection Summary', 'System Health Status'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"type": "bar"}, {"type": "indicator"}]]
        )
        
        # Temperature plot
        temp_cols = [col for col in sensor_cols if 'Temp' in col]
        for col in temp_cols[:2]:
            fig.add_trace(
                go.Scatter(
                    x=df.index[-100:],  # Last 100 points for overview
                    y=df[col].iloc[-100:],
                    mode='lines',
                    name=col,
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # Pressure plot
        draft_cols = [col for col in sensor_cols if 'draft' in col]
        for col in draft_cols[:2]:
            fig.add_trace(
                go.Scatter(
                    x=df.index[-100:],
                    y=df[col].iloc[-100:],
                    mode='lines',
                    name=col,
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Anomaly summary
        methods = list(anomaly_summary.keys())[:5]  # Limit to 5 methods
        rates = [anomaly_summary[method]['rate_percent'] for method in methods]
        
        fig.add_trace(
            go.Bar(
                x=methods,
                y=rates,
                name='Anomaly Rate',
                marker_color=self.color_palette['anomaly'],
                showlegend=False
            ),
            row=2, col=1
        )
        
        # System health indicator
        avg_anomaly_rate = np.mean(rates) if rates else 0
        health_score = max(0, 100 - avg_anomaly_rate)
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=health_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "System Health"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Cyclone System Overview Dashboard",
            height=600,
            showlegend=False
        )
        
        return fig
