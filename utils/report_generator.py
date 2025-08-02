import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st
from typing import Dict, List, Optional, Tuple
from io import BytesIO
import base64
import json

class ReportGenerator:
    """Generate automated PowerPoint reports for cyclone anomaly analysis."""
    
    def __init__(self):
        self.report_data = {}
    
    def generate_report_data(self, df: pd.DataFrame, anomaly_summary: Dict, 
                           anomaly_periods: List[Dict]) -> Dict:
        """
        Generate comprehensive report data.
        
        Args:
            df: Analyzed DataFrame
            anomaly_summary: Summary of anomaly detection results
            anomaly_periods: List of detected anomaly periods
            
        Returns:
            Dictionary with report data
        """
        # Basic statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Data quality metrics
        data_quality = {
            'total_records': len(df),
            'missing_data_percent': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'date_range': {
                'start': str(df.index.min()) if hasattr(df.index, 'min') else 'N/A',
                'end': str(df.index.max()) if hasattr(df.index, 'max') else 'N/A',
                'duration_days': (df.index.max() - df.index.min()).days if hasattr(df.index, 'min') else 0
            }
        }
        
        # Variable statistics
        variable_stats = {}
        for col in numeric_cols:
            variable_stats[col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'median': float(df[col].median())
            }
        
        # Correlation analysis
        correlation_matrix = df[numeric_cols].corr()
        high_correlations = []
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_correlations.append({
                        'var1': numeric_cols[i],
                        'var2': numeric_cols[j],
                        'correlation': float(corr_val)
                    })
        
        # Anomaly analysis
        total_anomalies = sum([summary['count'] for summary in anomaly_summary.values()])
        anomaly_rate = (total_anomalies / len(df)) * 100 if len(df) > 0 else 0
        
        # Most critical periods
        if anomaly_periods:
            critical_periods = sorted(anomaly_periods, 
                                    key=lambda x: x.get('severity_score', 0), 
                                    reverse=True)[:5]
        else:
            critical_periods = []
        
        # Key insights
        insights = self._generate_insights(df, anomaly_summary, anomaly_periods, variable_stats)
        
        report_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'data_quality': data_quality,
                'analysis_methods': list(anomaly_summary.keys())
            },
            'data_preparation': {
                'variables': numeric_cols,
                'statistics': variable_stats,
                'correlations': high_correlations
            },
            'analysis_strategy': {
                'methods_used': list(anomaly_summary.keys()),
                'detection_parameters': self._get_detection_parameters(),
                'data_preprocessing': self._get_preprocessing_steps()
            },
            'key_insights': {
                'total_anomalies': int(total_anomalies),
                'anomaly_rate_percent': round(anomaly_rate, 2),
                'anomaly_periods_count': len(anomaly_periods),
                'critical_periods': critical_periods,
                'method_comparison': anomaly_summary,
                'insights': insights
            },
            'recommendations': self._generate_recommendations(anomaly_summary, anomaly_periods)
        }
        
        self.report_data = report_data
        return report_data
    
    def _generate_insights(self, df: pd.DataFrame, anomaly_summary: Dict, 
                          anomaly_periods: List[Dict], variable_stats: Dict) -> List[str]:
        """Generate key insights from the analysis."""
        insights = []
        
        # Most anomalous method
        if anomaly_summary:
            most_sensitive = max(anomaly_summary.keys(), 
                               key=lambda k: anomaly_summary[k]['rate_percent'])
            insights.append(f"The {most_sensitive} method detected the highest anomaly rate "
                          f"({anomaly_summary[most_sensitive]['rate_percent']:.1f}%)")
        
        # Variable with highest variability
        if variable_stats:
            cv_scores = {var: stats['std'] / stats['mean'] if stats['mean'] != 0 else 0 
                        for var, stats in variable_stats.items()}
            most_variable = max(cv_scores.keys(), key=lambda k: cv_scores[k])
            insights.append(f"{most_variable} shows the highest variability "
                          f"(CV: {cv_scores[most_variable]:.2f})")
        
        # Anomaly period patterns
        if anomaly_periods:
            avg_duration = np.mean([p['duration_hours'] for p in anomaly_periods])
            insights.append(f"Average anomaly period duration: {avg_duration:.1f} hours")
            
            if len(anomaly_periods) > 1:
                durations = [p['duration_hours'] for p in anomaly_periods]
                if max(durations) > 2 * np.mean(durations):
                    insights.append("Some anomaly periods are significantly longer than others, "
                                  "indicating potential system failures")
        
        # Temperature vs Draft relationship
        temp_cols = [col for col in df.columns if 'Temp' in col]
        draft_cols = [col for col in df.columns if 'draft' in col]
        if temp_cols and draft_cols:
            corr = df[temp_cols[0]].corr(df[draft_cols[0]])
            if abs(corr) > 0.5:
                insights.append(f"Strong correlation ({corr:.2f}) between temperature and draft "
                              "indicates coupled system behavior")
        
        return insights
    
    def _get_detection_parameters(self) -> Dict:
        """Get detection parameters used in analysis."""
        return {
            'zscore_threshold': 3.0,
            'iqr_multiplier': 1.5,
            'isolation_forest_contamination': 0.1,
            'statistical_window_size': 24,
            'statistical_threshold': 2.0
        }
    
    def _get_preprocessing_steps(self) -> List[str]:
        """Get preprocessing steps applied to data."""
        return [
            "DateTime parsing and indexing",
            "Missing value imputation using forward/backward fill",
            "Outlier detection using IQR method",
            "Data normalization for ML algorithms",
            "Rolling statistics calculation"
        ]
    
    def _generate_recommendations(self, anomaly_summary: Dict, 
                                anomaly_periods: List[Dict]) -> List[str]:
        """Generate operational recommendations."""
        recommendations = []
        
        # Based on anomaly rate
        total_rate = sum([summary['rate_percent'] for summary in anomaly_summary.values()]) / len(anomaly_summary)
        
        if total_rate > 10:
            recommendations.append("High anomaly rate detected (>10%). Consider immediate system inspection.")
        elif total_rate > 5:
            recommendations.append("Moderate anomaly rate (5-10%). Schedule preventive maintenance.")
        else:
            recommendations.append("Low anomaly rate (<5%). System operating within normal parameters.")
        
        # Based on anomaly periods
        if anomaly_periods:
            long_periods = [p for p in anomaly_periods if p['duration_hours'] > 24]
            if long_periods:
                recommendations.append(f"Detected {len(long_periods)} extended anomaly periods (>24h). "
                                     "Investigate root causes.")
        
        # Method-specific recommendations
        if 'isolation_forest' in anomaly_summary:
            if anomaly_summary['isolation_forest']['rate_percent'] > 15:
                recommendations.append("Isolation Forest detected high multivariate anomalies. "
                                     "Check sensor calibration and system integration.")
        
        if 'correlation' in anomaly_summary:
            if anomaly_summary['correlation']['rate_percent'] > 5:
                recommendations.append("Correlation breakdowns detected. Monitor sensor relationships "
                                     "and system coupling.")
        
        # General recommendations
        recommendations.extend([
            "Implement real-time monitoring for early anomaly detection",
            "Establish automated alerts for critical anomaly thresholds",
            "Regular sensor calibration and maintenance scheduling",
            "Historical trend analysis for predictive maintenance"
        ])
        
        return recommendations
    
    def export_to_json(self) -> str:
        """Export report data to JSON format."""
        return json.dumps(self.report_data, indent=2, default=str)
    
    def export_to_csv(self, df: pd.DataFrame, anomaly_periods: List[Dict]) -> Tuple[str, str]:
        """
        Export analysis results to CSV format.
        
        Returns:
            Tuple of (summary_csv, periods_csv)
        """
        # Summary CSV
        summary_data = []
        if 'key_insights' in self.report_data:
            method_comparison = self.report_data['key_insights']['method_comparison']
            for method, stats in method_comparison.items():
                summary_data.append({
                    'Method': method,
                    'Anomaly_Count': stats['count'],
                    'Anomaly_Rate_Percent': stats['rate_percent'],
                    'First_Occurrence': stats['first_occurrence'],
                    'Last_Occurrence': stats['last_occurrence']
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_csv = summary_df.to_csv(index=False)
        
        # Periods CSV
        periods_df = pd.DataFrame(anomaly_periods)
        periods_csv = periods_df.to_csv(index=False)
        
        return summary_csv, periods_csv
    
    def generate_html_report(self) -> str:
        """Generate HTML report."""
        if not self.report_data:
            return "<html><body><h1>No report data available</h1></body></html>"
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Cyclone Anomaly Detection Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f2f6; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 30px 0; }}
                .metric {{ background-color: #e8f4f8; padding: 10px; margin: 10px 0; border-radius: 3px; }}
                .insight {{ background-color: #fff3cd; padding: 10px; margin: 5px 0; border-radius: 3px; }}
                .recommendation {{ background-color: #d4edda; padding: 10px; margin: 5px 0; border-radius: 3px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🌪️ Cyclone Anomaly Detection Report</h1>
                <p><strong>Generated:</strong> {self.report_data['metadata']['generated_at']}</p>
                <p><strong>Data Period:</strong> {self.report_data['metadata']['data_quality']['date_range']['start']} 
                   to {self.report_data['metadata']['data_quality']['date_range']['end']}</p>
            </div>
            
            <div class="section">
                <h2>📊 Data Overview</h2>
                <div class="metric">
                    <strong>Total Records:</strong> {self.report_data['metadata']['data_quality']['total_records']:,}
                </div>
                <div class="metric">
                    <strong>Duration:</strong> {self.report_data['metadata']['data_quality']['date_range']['duration_days']} days
                </div>
                <div class="metric">
                    <strong>Missing Data:</strong> {self.report_data['metadata']['data_quality']['missing_data_percent']:.2f}%
                </div>
            </div>
            
            <div class="section">
                <h2>🎯 Key Findings</h2>
                <div class="metric">
                    <strong>Total Anomalies Detected:</strong> {self.report_data['key_insights']['total_anomalies']:,}
                </div>
                <div class="metric">
                    <strong>Overall Anomaly Rate:</strong> {self.report_data['key_insights']['anomaly_rate_percent']:.2f}%
                </div>
                <div class="metric">
                    <strong>Anomaly Periods:</strong> {self.report_data['key_insights']['anomaly_periods_count']}
                </div>
                
                <h3>Key Insights</h3>
        """
        
        for insight in self.report_data['key_insights']['insights']:
            html += f'<div class="insight">• {insight}</div>'
        
        html += """
            </div>
            
            <div class="section">
                <h2>📋 Method Comparison</h2>
                <table>
                    <tr>
                        <th>Method</th>
                        <th>Anomalies Detected</th>
                        <th>Detection Rate (%)</th>
                        <th>First Occurrence</th>
                    </tr>
        """
        
        for method, stats in self.report_data['key_insights']['method_comparison'].items():
            html += f"""
                <tr>
                    <td>{method.title()}</td>
                    <td>{stats['count']:,}</td>
                    <td>{stats['rate_percent']:.2f}%</td>
                    <td>{stats['first_occurrence'] or 'N/A'}</td>
                </tr>
            """
        
        html += """
                </table>
            </div>
            
            <div class="section">
                <h2>💡 Recommendations</h2>
        """
        
        for rec in self.report_data['recommendations']:
            html += f'<div class="recommendation">• {rec}</div>'
        
        html += """
            </div>
        </body>
        </html>
        """
        
        return html
    
    def create_powerpoint_content(self) -> Dict:
        """
        Create PowerPoint content structure (to be used with python-pptx if available).
        
        Returns:
            Dictionary with slide content
        """
        if not self.report_data:
            return {}
        
        slides = {
            'slide_1': {
                'title': 'Cyclone Anomaly Detection Analysis',
                'subtitle': f"Analysis Period: {self.report_data['metadata']['data_quality']['date_range']['start']} to {self.report_data['metadata']['data_quality']['date_range']['end']}",
                'content': [
                    f"📊 {self.report_data['metadata']['data_quality']['total_records']:,} records analyzed",
                    f"🎯 {len(self.report_data['metadata']['analysis_methods'])} detection methods used",
                    f"⚠️ {self.report_data['key_insights']['total_anomalies']:,} anomalies detected ({self.report_data['key_insights']['anomaly_rate_percent']:.1f}%)"
                ]
            },
            'slide_2': {
                'title': 'Data Preparation & Quality',
                'content': [
                    f"✅ Data Quality Score: {100 - self.report_data['metadata']['data_quality']['missing_data_percent']:.1f}%",
                    f"📅 Duration: {self.report_data['metadata']['data_quality']['date_range']['duration_days']} days",
                    f"🔧 Variables Analyzed: {len(self.report_data['data_preparation']['variables'])}",
                    "📈 Preprocessing Applied:",
                    *[f"  • {step}" for step in self.report_data['analysis_strategy']['data_preprocessing'][:3]]
                ]
            },
            'slide_3': {
                'title': 'Analysis Strategy & Methods',
                'content': [
                    "🔍 Anomaly Detection Methods:",
                    *[f"  • {method.title().replace('_', ' ')}" for method in self.report_data['analysis_strategy']['methods_used']],
                    "",
                    "⚙️ Key Parameters:",
                    "  • Z-Score Threshold: 3.0",
                    "  • IQR Multiplier: 1.5",
                    "  • Rolling Window: 24 periods"
                ]
            },
            'slide_4': {
                'title': 'Key Insights & Detected Periods',
                'content': [
                    f"🚨 {self.report_data['key_insights']['anomaly_periods_count']} anomaly periods identified",
                    f"📊 Average anomaly rate: {self.report_data['key_insights']['anomaly_rate_percent']:.1f}%",
                    "",
                    "🔍 Key Insights:",
                    *[f"  • {insight}" for insight in self.report_data['key_insights']['insights'][:4]]
                ]
            },
            'slide_5': {
                'title': 'Recommendations & Next Steps',
                'content': [
                    "💡 Operational Recommendations:",
                    *[f"  • {rec}" for rec in self.report_data['recommendations'][:6]]
                ]
            }
        }
        
        return slides
