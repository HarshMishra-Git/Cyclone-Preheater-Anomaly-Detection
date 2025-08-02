import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.data_processor import DataProcessor
from utils.visualizer import Visualizer

st.set_page_config(
    page_title="Data Overview - Cyclone Analysis",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Data Overview & Quality Assessment")

# Check if data is loaded
if not st.session_state.get('data_loaded', False) or st.session_state.get('processed_data') is None:
    st.warning("⚠️ Please load data first from the main page.")
    st.stop()

df = st.session_state.processed_data
processor = DataProcessor()
visualizer = Visualizer()

# Sidebar controls
st.sidebar.header("📋 Analysis Controls")
show_raw_data = st.sidebar.checkbox("Show Raw Data Sample", value=False)
sample_size = st.sidebar.slider("Sample Size for Quick Analysis", 1000, 50000, 10000)

# Data quality report
with st.spinner("Generating data quality report..."):
    quality_report = processor.get_data_quality_report(df)

# Overview metrics
st.header("📈 Dataset Summary")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Records", f"{quality_report['total_records']:,}")

with col2:
    duration = quality_report['date_range']['end'] - quality_report['date_range']['start']
    st.metric("Duration", f"{duration.days} days")

with col3:
    st.metric("Variables", quality_report['total_variables'])

with col4:
    missing_pct = sum(quality_report['missing_percentage'].values()) / len(quality_report['missing_percentage'])
    st.metric("Avg Missing %", f"{missing_pct:.2f}%")

with col5:
    memory_mb = quality_report['memory_usage'] / (1024 * 1024)
    st.metric("Memory Usage", f"{memory_mb:.1f} MB")

st.markdown("---")

# Data quality details
st.header("🔍 Data Quality Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Missing Data by Variable")
    missing_data = pd.DataFrame({
        'Variable': list(quality_report['missing_values'].keys()),
        'Missing_Count': list(quality_report['missing_values'].values()),
        'Missing_Percentage': list(quality_report['missing_percentage'].values())
    })
    
    fig_missing = px.bar(
        missing_data, 
        x='Variable', 
        y='Missing_Percentage',
        title="Missing Data Percentage by Variable",
        color='Missing_Percentage',
        color_continuous_scale='Reds'
    )
    fig_missing.update_xaxes(tickangle=45)
    st.plotly_chart(fig_missing, use_container_width=True)

with col2:
    st.subheader("Data Types")
    dtype_info = pd.DataFrame({
        'Variable': list(quality_report['data_types'].keys()),
        'Data_Type': [str(dtype) for dtype in quality_report['data_types'].values()]
    })
    st.dataframe(dtype_info, hide_index=True, use_container_width=True)

# Statistical overview
st.header("📊 Statistical Overview")

# Get numeric columns for analysis
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Basic statistics
st.subheader("Descriptive Statistics")
stats_df = df[numeric_cols].describe()
st.dataframe(stats_df, use_container_width=True)

# Variable selection for detailed analysis
st.subheader("Variable Analysis")
selected_vars = st.multiselect(
    "Select variables for detailed analysis:",
    numeric_cols,
    default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
)

if selected_vars:
    # Create distribution plots
    fig_dist = make_subplots(
        rows=2, cols=2,
        subplot_titles=selected_vars[:4],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    for i, var in enumerate(selected_vars[:4]):
        row = (i // 2) + 1
        col = (i % 2) + 1
        
        fig_dist.add_trace(
            go.Histogram(x=df[var], name=var, showlegend=False),
            row=row, col=col
        )
    
    fig_dist.update_layout(
        title="Distribution Analysis",
        height=600
    )
    st.plotly_chart(fig_dist, use_container_width=True)

# Time series overview
st.header("⏱️ Time Series Overview")

# Sample data for performance
if len(df) > sample_size:
    df_sample = processor.sample_data(df, sample_size, method='systematic')
    st.info(f"Showing sample of {len(df_sample):,} records for performance")
else:
    df_sample = df

# Variable selector for time series
ts_var = st.selectbox(
    "Select variable for time series visualization:",
    numeric_cols,
    index=0
)

if ts_var:
    fig_ts = visualizer.create_time_series_plot(df_sample, [ts_var])
    st.plotly_chart(fig_ts, use_container_width=True)

# Correlation analysis
st.header("🔗 Correlation Analysis")

# Calculate correlation matrix
corr_matrix = df[selected_vars].corr() if selected_vars else df[numeric_cols].corr()

# Create correlation heatmap
fig_corr = visualizer.create_correlation_heatmap(df, selected_vars if selected_vars else numeric_cols)
st.plotly_chart(fig_corr, use_container_width=True)

# High correlations table
st.subheader("High Correlations (|r| > 0.7)")
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        corr_val = corr_matrix.iloc[i, j]
        if abs(corr_val) > 0.7:
            high_corr_pairs.append({
                'Variable 1': corr_matrix.columns[i],
                'Variable 2': corr_matrix.columns[j],
                'Correlation': round(corr_val, 3)
            })

if high_corr_pairs:
    high_corr_df = pd.DataFrame(high_corr_pairs)
    st.dataframe(high_corr_df, hide_index=True, use_container_width=True)
else:
    st.info("No high correlations (|r| > 0.7) found between selected variables.")

# Data quality insights
st.header("💡 Data Quality Insights")

insights = []

# Missing data insights
total_missing = sum(quality_report['missing_values'].values())
if total_missing == 0:
    insights.append("✅ No missing values detected - excellent data quality")
elif total_missing < quality_report['total_records'] * 0.01:
    insights.append("✅ Very low missing data rate (<1%) - good data quality")
else:
    insights.append(f"⚠️ {total_missing:,} missing values ({total_missing/quality_report['total_records']*100:.1f}%) - consider data cleaning")

# Duplicates insight
if quality_report['duplicated_rows'] == 0:
    insights.append("✅ No duplicate records found")
else:
    insights.append(f"⚠️ {quality_report['duplicated_rows']} duplicate records detected")

# Data frequency insight
if quality_report['date_range']['frequency']:
    insights.append(f"📅 Data frequency: {quality_report['date_range']['frequency']}")

# Variable relationships
if high_corr_pairs:
    insights.append(f"🔗 {len(high_corr_pairs)} high correlation pairs detected - variables show strong relationships")

for insight in insights:
    st.info(insight)

# Raw data preview
if show_raw_data:
    st.header("🔍 Raw Data Sample")
    st.subheader("First 100 Records")
    st.dataframe(df.head(100), use_container_width=True)
    
    st.subheader("Last 100 Records")
    st.dataframe(df.tail(100), use_container_width=True)

# Export section
st.header("📥 Export Options")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Export Statistics"):
        stats_csv = stats_df.to_csv()
        st.download_button(
            label="Download Statistics CSV",
            data=stats_csv,
            file_name="cyclone_statistics.csv",
            mime="text/csv"
        )

with col2:
    if st.button("🔗 Export Correlations"):
        corr_csv = corr_matrix.to_csv()
        st.download_button(
            label="Download Correlation Matrix CSV",
            data=corr_csv,
            file_name="cyclone_correlations.csv",
            mime="text/csv"
        )

with col3:
    if st.button("📋 Export Quality Report"):
        import json
        quality_json = json.dumps(quality_report, indent=2, default=str)
        st.download_button(
            label="Download Quality Report JSON",
            data=quality_json,
            file_name="data_quality_report.json",
            mime="application/json"
        )
