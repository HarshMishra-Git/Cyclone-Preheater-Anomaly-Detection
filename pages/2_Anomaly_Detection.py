import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.data_processor import DataProcessor
from utils.anomaly_detector import AnomalyDetector
from utils.visualizer import Visualizer

st.set_page_config(
    page_title="Anomaly Detection - Cyclone Analysis",
    page_icon="⚠️",
    layout="wide"
)

st.title("⚠️ Anomaly Detection Analysis")

# Check if data is loaded
if not st.session_state.get('data_loaded', False) or st.session_state.get('processed_data') is None:
    st.warning("⚠️ Please load data first from the main page.")
    st.stop()

df = st.session_state.processed_data
detector = AnomalyDetector()
visualizer = Visualizer()
processor = DataProcessor()

# Initialize session state for analysis results
if 'anomaly_results' not in st.session_state:
    st.session_state.anomaly_results = {}
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

# Sidebar controls
st.sidebar.header("🔧 Detection Parameters")

# Method selection
st.sidebar.subheader("Detection Methods")
use_zscore = st.sidebar.checkbox("Z-Score Detection", value=True)
use_iqr = st.sidebar.checkbox("IQR Detection", value=True)
use_isolation_forest = st.sidebar.checkbox("Isolation Forest", value=True)
use_statistical = st.sidebar.checkbox("Statistical Anomalies", value=True)

# Parameters
st.sidebar.subheader("Method Parameters")

zscore_threshold = st.sidebar.slider("Z-Score Threshold", 2.0, 4.0, 3.0, 0.1)
iqr_multiplier = st.sidebar.slider("IQR Multiplier", 1.0, 3.0, 1.5, 0.1)
isolation_contamination = st.sidebar.slider("Isolation Forest Contamination", 0.01, 0.2, 0.1, 0.01)
statistical_window = st.sidebar.slider("Statistical Window Size", 12, 72, 24, 6)
statistical_threshold = st.sidebar.slider("Statistical Threshold", 1.5, 3.0, 2.0, 0.1)

# Variable selection
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
selected_columns = st.sidebar.multiselect(
    "Select Variables for Analysis",
    numeric_cols,
    default=numeric_cols
)

# Sampling for performance
sample_size = st.sidebar.slider("Sample Size (for performance)", 5000, 50000, 20000, 5000)
use_sampling = st.sidebar.checkbox("Use Sampling for Large Dataset", value=len(df) > 50000)

# Analysis execution
st.sidebar.markdown("---")
run_analysis = st.sidebar.button("🚀 Run Anomaly Detection", type="primary")

# Main content
if run_analysis and selected_columns:
    # Prepare data
    if use_sampling and len(df) > sample_size:
        df_analysis = processor.sample_data(df, sample_size, method='systematic')
        st.info(f"Using sample of {len(df_analysis):,} records for analysis")
    else:
        df_analysis = df.copy()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = df_analysis.copy()
    
    # Run selected detection methods
    methods_run = []
    progress = 0
    total_methods = sum([use_zscore, use_iqr, use_isolation_forest, use_statistical])
    
    if use_zscore:
        status_text.text("Running Z-Score Detection...")
        results = detector.detect_zscore_anomalies(results, zscore_threshold, selected_columns)
        methods_run.append('zscore')
        progress += 1
        progress_bar.progress(progress / total_methods)
    
    if use_iqr:
        status_text.text("Running IQR Detection...")
        results = detector.detect_iqr_anomalies(results, iqr_multiplier, selected_columns)
        methods_run.append('iqr')
        progress += 1
        progress_bar.progress(progress / total_methods)
    
    if use_isolation_forest:
        status_text.text("Running Isolation Forest...")
        results = detector.detect_isolation_forest_anomalies(
            results, contamination=isolation_contamination, columns=selected_columns
        )
        methods_run.append('isolation_forest')
        progress += 1
        progress_bar.progress(progress / total_methods)
    
    if use_statistical:
        status_text.text("Running Statistical Detection...")
        results = detector.detect_statistical_anomalies(
            results, statistical_window, statistical_threshold, selected_columns
        )
        methods_run.append('statistical')
        progress += 1
        progress_bar.progress(progress / total_methods)
    
    # Combine methods
    status_text.text("Combining detection methods...")
    results = detector.combine_anomaly_methods(results, methods_run)
    
    # Generate summary
    anomaly_summary = detector.get_anomaly_summary(results)
    anomaly_periods = detector.get_anomaly_periods(results, 'combined', min_duration_hours=1)
    
    # Store results
    st.session_state.anomaly_results = {
        'data': results,
        'summary': anomaly_summary,
        'periods': anomaly_periods,
        'methods': methods_run,
        'parameters': {
            'zscore_threshold': zscore_threshold,
            'iqr_multiplier': iqr_multiplier,
            'isolation_contamination': isolation_contamination,
            'statistical_window': statistical_window,
            'statistical_threshold': statistical_threshold
        }
    }
    st.session_state.analysis_complete = True
    
    progress_bar.progress(1.0)
    status_text.text("Analysis complete! ✅")
    
    # Auto-refresh to show results
    st.rerun()

# Display results if analysis is complete
if st.session_state.analysis_complete and st.session_state.anomaly_results:
    results = st.session_state.anomaly_results['data']
    anomaly_summary = st.session_state.anomaly_results['summary']
    anomaly_periods = st.session_state.anomaly_results['periods']
    methods_run = st.session_state.anomaly_results['methods']
    
    # Results overview
    st.header("📊 Detection Results Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_anomalies = sum([summary['count'] for summary in anomaly_summary.values()])
    anomaly_rate = (total_anomalies / len(results)) * 100 if len(results) > 0 else 0
    
    with col1:
        st.metric("Total Anomalies", f"{total_anomalies:,}")
    
    with col2:
        st.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")
    
    with col3:
        st.metric("Anomaly Periods", len(anomaly_periods))
    
    with col4:
        avg_severity = np.mean([p.get('severity_score', 0) for p in anomaly_periods]) if anomaly_periods else 0
        st.metric("Avg Severity", f"{avg_severity:.2f}")
    
    # Method comparison
    st.subheader("🔍 Method Comparison")
    fig_summary = visualizer.create_anomaly_summary_chart(anomaly_summary)
    st.plotly_chart(fig_summary, use_container_width=True)
    
    # Detailed method results
    st.subheader("📋 Detailed Results by Method")
    
    summary_df = pd.DataFrame([
        {
            'Method': method,
            'Anomalies': stats['count'],
            'Rate (%)': stats['rate_percent'],
            'First Occurrence': stats['first_occurrence'],
            'Last Occurrence': stats['last_occurrence']
        }
        for method, stats in anomaly_summary.items()
    ])
    st.dataframe(summary_df, hide_index=True, use_container_width=True)
    
    # Visualization section
    st.header("📈 Anomaly Visualizations")
    
    # Variable selector for visualization
    viz_variable = st.selectbox(
        "Select variable for anomaly visualization:",
        selected_columns,
        key="viz_var"
    )
    
    # Method selector for visualization
    viz_method = st.selectbox(
        "Select detection method to highlight:",
        ['combined'] + methods_run,
        key="viz_method"
    )
    
    if viz_variable and viz_method:
        # Create anomaly plot
        anomaly_col = f'anomaly_{viz_method}'
        fig_anomaly = visualizer.create_anomaly_plot(
            results, viz_variable, anomaly_col,
            title=f"{viz_variable} - {viz_method.title()} Anomalies"
        )
        st.plotly_chart(fig_anomaly, use_container_width=True)
    
    # Anomaly score visualization
    st.subheader("🎯 Anomaly Scores Comparison")
    
    score_columns = [col for col in results.columns if col.startswith('anomaly_score_')]
    if score_columns:
        fig_scores = visualizer.create_anomaly_score_plot(results, score_columns)
        st.plotly_chart(fig_scores, use_container_width=True)
    
    # Anomaly periods timeline
    if anomaly_periods:
        st.subheader("⏰ Anomaly Periods Timeline")
        fig_timeline = visualizer.create_anomaly_timeline(
            anomaly_periods,
            title="Detected Anomaly Periods"
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Periods table
        st.subheader("📅 Anomaly Periods Details")
        periods_df = pd.DataFrame(anomaly_periods)
        if not periods_df.empty:
            periods_df['duration_hours'] = periods_df['duration_hours'].round(2)
            periods_df['severity_score'] = periods_df['severity_score'].round(3)
            st.dataframe(periods_df, hide_index=True, use_container_width=True)
    
    # Multivariate analysis
    if len(selected_columns) >= 3:
        st.subheader("🌐 Multivariate Anomaly Analysis")
        
        # Select 3 variables for 3D plot
        mv_vars = st.multiselect(
            "Select 3 variables for multivariate visualization:",
            selected_columns,
            default=selected_columns[:3],
            max_selections=3,
            key="mv_vars"
        )
        
        if len(mv_vars) >= 2:
            fig_3d = visualizer.create_multivariate_plot(
                results, mv_vars, 'anomaly_combined',
                title="Multivariate Anomaly Detection"
            )
            st.plotly_chart(fig_3d, use_container_width=True)
    
    # Statistical insights
    st.header("💡 Key Insights")
    
    insights = []
    
    # Method effectiveness
    if anomaly_summary:
        most_sensitive = max(anomaly_summary.keys(), key=lambda k: anomaly_summary[k]['rate_percent'])
        least_sensitive = min(anomaly_summary.keys(), key=lambda k: anomaly_summary[k]['rate_percent'])
        
        insights.append(f"🔍 **Most Sensitive Method**: {most_sensitive.title()} detected {anomaly_summary[most_sensitive]['rate_percent']:.1f}% anomalies")
        insights.append(f"🔍 **Least Sensitive Method**: {least_sensitive.title()} detected {anomaly_summary[least_sensitive]['rate_percent']:.1f}% anomalies")
    
    # Temporal patterns
    if anomaly_periods:
        durations = [p['duration_hours'] for p in anomaly_periods]
        avg_duration = np.mean(durations)
        max_duration = max(durations)
        
        insights.append(f"⏱️ **Average Anomaly Duration**: {avg_duration:.1f} hours")
        insights.append(f"⏱️ **Longest Anomaly Period**: {max_duration:.1f} hours")
        
        # Check for recent anomalies
        recent_periods = [p for p in anomaly_periods 
                         if p['end_time'] >= (results.index.max() - timedelta(days=7))]
        if recent_periods:
            insights.append(f"🚨 **Recent Activity**: {len(recent_periods)} anomaly periods in the last 7 days")
    
    # System health assessment
    if anomaly_rate < 1:
        insights.append("✅ **System Health**: Excellent - Very low anomaly rate (<1%)")
    elif anomaly_rate < 5:
        insights.append("⚠️ **System Health**: Good - Low anomaly rate (1-5%)")
    elif anomaly_rate < 10:
        insights.append("⚠️ **System Health**: Moderate - Medium anomaly rate (5-10%)")
    else:
        insights.append("🚨 **System Health**: Poor - High anomaly rate (>10%)")
    
    for insight in insights:
        st.info(insight)
    
    # Export section
    st.header("📥 Export Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 Export Anomaly Data"):
            anomaly_csv = results.to_csv()
            st.download_button(
                label="Download Anomaly Data CSV",
                data=anomaly_csv,
                file_name=f"anomaly_detection_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📋 Export Summary"):
            summary_csv = summary_df.to_csv(index=False)
            st.download_button(
                label="Download Summary CSV",
                data=summary_csv,
                file_name=f"anomaly_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("📅 Export Periods"):
            if anomaly_periods:
                periods_csv = pd.DataFrame(anomaly_periods).to_csv(index=False)
                st.download_button(
                    label="Download Periods CSV",
                    data=periods_csv,
                    file_name=f"anomaly_periods_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No anomaly periods to export")
    
    with col4:
        if st.button("🔧 Export Parameters"):
            import json
            params_json = json.dumps(st.session_state.anomaly_results['parameters'], indent=2)
            st.download_button(
                label="Download Parameters JSON",
                data=params_json,
                file_name=f"detection_parameters_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json"
            )

else:
    # Initial instructions
    st.info("""
    ## 🚀 Getting Started with Anomaly Detection
    
    1. **Select Variables**: Choose which sensor variables to analyze
    2. **Configure Methods**: Enable/disable detection methods and adjust parameters
    3. **Set Sampling**: Use sampling for large datasets to improve performance
    4. **Run Analysis**: Click the "Run Anomaly Detection" button
    
    ### Available Detection Methods:
    
    - **Z-Score**: Detects statistical outliers based on standard deviations
    - **IQR**: Uses interquartile range to identify outliers
    - **Isolation Forest**: ML-based multivariate anomaly detection
    - **Statistical**: Rolling window statistical analysis
    
    ### Performance Tips:
    - Use sampling for datasets >50K records
    - Start with fewer variables for initial exploration
    - Adjust thresholds based on domain knowledge
    """)
    
    # Quick stats about the data
    st.subheader("📊 Data Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    
    with col2:
        st.metric("Available Variables", len(numeric_cols))
    
    with col3:
        duration = df.index.max() - df.index.min()
        st.metric("Data Duration", f"{duration.days} days")
