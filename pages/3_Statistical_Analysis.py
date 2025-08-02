import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from scipy import stats
from scipy.stats import normaltest, shapiro, anderson
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
import warnings
warnings.filterwarnings('ignore')

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.data_processor import DataProcessor
from utils.visualizer import Visualizer

st.set_page_config(
    page_title="Statistical Analysis - Cyclone Analysis",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Advanced Statistical Analysis")

# Check if data is loaded
if not st.session_state.get('data_loaded', False) or st.session_state.get('processed_data') is None:
    st.warning("⚠️ Please load data first from the main page.")
    st.stop()

df = st.session_state.processed_data
processor = DataProcessor()
visualizer = Visualizer()

# Sidebar controls
st.sidebar.header("📊 Analysis Settings")

# Variable selection
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
selected_vars = st.sidebar.multiselect(
    "Select Variables for Analysis",
    numeric_cols,
    default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
)

# Analysis type selection
analysis_types = st.sidebar.multiselect(
    "Select Analysis Types",
    ["Correlation Analysis", "Normality Tests", "Time Series Decomposition", 
     "Stationarity Tests", "Trend Analysis", "Statistical Relationships"],
    default=["Correlation Analysis", "Time Series Decomposition"]
)

# Sampling for performance
sample_size = st.sidebar.slider("Sample Size", 5000, min(50000, len(df)), 20000, 5000)
use_sampling = st.sidebar.checkbox("Use Sampling", value=len(df) > 30000)

# Time window for analysis
if hasattr(df.index, 'min'):
    start_date = st.sidebar.date_input("Analysis Start Date", value=df.index.min().date())
    end_date = st.sidebar.date_input("Analysis End Date", value=df.index.max().date())
    
    # Filter data by date range
    df_filtered = df[(df.index.date >= start_date) & (df.index.date <= end_date)]
else:
    df_filtered = df

# Apply sampling if needed
if use_sampling and len(df_filtered) > sample_size:
    df_analysis = processor.sample_data(df_filtered, sample_size, method='systematic')
    st.info(f"Analysis using {len(df_analysis):,} sampled records")
else:
    df_analysis = df_filtered
    st.info(f"Analysis using {len(df_analysis):,} records")

if not selected_vars:
    st.warning("Please select at least one variable for analysis.")
    st.stop()

# Main analysis content
st.header("📊 Statistical Analysis Results")

# Correlation Analysis
if "Correlation Analysis" in analysis_types:
    st.subheader("🔗 Advanced Correlation Analysis")
    
    # Calculate different correlation types
    pearson_corr = df_analysis[selected_vars].corr(method='pearson')
    spearman_corr = df_analysis[selected_vars].corr(method='spearman')
    kendall_corr = df_analysis[selected_vars].corr(method='kendall')
    
    # Create tabs for different correlation types
    corr_tab1, corr_tab2, corr_tab3 = st.tabs(["Pearson", "Spearman", "Kendall"])
    
    with corr_tab1:
        st.write("**Pearson Correlation** (Linear relationships)")
        fig_pearson = visualizer.create_correlation_heatmap(
            df_analysis, selected_vars, "Pearson Correlation Matrix"
        )
        st.plotly_chart(fig_pearson, use_container_width=True)
    
    with corr_tab2:
        st.write("**Spearman Correlation** (Monotonic relationships)")
        fig_spearman = go.Figure(data=go.Heatmap(
            z=spearman_corr.values,
            x=spearman_corr.columns,
            y=spearman_corr.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(spearman_corr.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        fig_spearman.update_layout(title="Spearman Correlation Matrix", height=500)
        st.plotly_chart(fig_spearman, use_container_width=True)
    
    with corr_tab3:
        st.write("**Kendall Correlation** (Rank-based relationships)")
        fig_kendall = go.Figure(data=go.Heatmap(
            z=kendall_corr.values,
            x=kendall_corr.columns,
            y=kendall_corr.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(kendall_corr.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        fig_kendall.update_layout(title="Kendall Correlation Matrix", height=500)
        st.plotly_chart(fig_kendall, use_container_width=True)
    
    # Correlation strength analysis
    st.subheader("🎯 Correlation Strength Analysis")
    
    correlation_strength = []
    for i in range(len(selected_vars)):
        for j in range(i + 1, len(selected_vars)):
            var1, var2 = selected_vars[i], selected_vars[j]
            pearson_val = pearson_corr.iloc[i, j]
            spearman_val = spearman_corr.iloc[i, j]
            kendall_val = kendall_corr.iloc[i, j]
            
            correlation_strength.append({
                'Variable Pair': f"{var1} - {var2}",
                'Pearson': round(pearson_val, 3),
                'Spearman': round(spearman_val, 3),
                'Kendall': round(kendall_val, 3),
                'Avg Strength': round(np.mean([abs(pearson_val), abs(spearman_val), abs(kendall_val)]), 3)
            })
    
    corr_strength_df = pd.DataFrame(correlation_strength)
    corr_strength_df = corr_strength_df.sort_values('Avg Strength', ascending=False)
    st.dataframe(corr_strength_df, hide_index=True, use_container_width=True)

# Normality Tests
if "Normality Tests" in analysis_types:
    st.subheader("📊 Normality Testing")
    
    normality_results = []
    
    for var in selected_vars:
        data = df_analysis[var].dropna()
        
        # Shapiro-Wilk test (for smaller samples)
        if len(data) <= 5000:
            shapiro_stat, shapiro_p = shapiro(data)
        else:
            shapiro_stat, shapiro_p = None, None
        
        # D'Agostino's normality test
        dagostino_stat, dagostino_p = normaltest(data)
        
        # Anderson-Darling test
        anderson_result = anderson(data, dist='norm')
        anderson_stat = anderson_result.statistic
        anderson_critical = anderson_result.critical_values[2]  # 5% significance level
        anderson_normal = anderson_stat < anderson_critical
        
        normality_results.append({
            'Variable': var,
            'Shapiro-Wilk Stat': round(shapiro_stat, 4) if shapiro_stat else 'N/A (sample too large)',
            'Shapiro-Wilk p-value': round(shapiro_p, 4) if shapiro_p else 'N/A',
            'D\'Agostino Stat': round(dagostino_stat, 4),
            'D\'Agostino p-value': round(dagostino_p, 4),
            'Anderson-Darling Stat': round(anderson_stat, 4),
            'Anderson Normal': 'Yes' if anderson_normal else 'No',
            'Overall Assessment': 'Normal' if (dagostino_p > 0.05 and anderson_normal) else 'Non-Normal'
        })
    
    normality_df = pd.DataFrame(normality_results)
    st.dataframe(normality_df, hide_index=True, use_container_width=True)
    
    st.info("""
    **Interpretation:**
    - **Shapiro-Wilk**: p > 0.05 suggests normal distribution
    - **D'Agostino**: p > 0.05 suggests normal distribution  
    - **Anderson-Darling**: Statistic below critical value suggests normal distribution
    """)
    
    # Distribution plots with normality overlay
    st.subheader("📈 Distribution Analysis")
    
    selected_var_dist = st.selectbox("Select variable for distribution analysis:", selected_vars, key="dist_var")
    
    fig_dist = make_subplots(rows=1, cols=2, subplot_titles=('Histogram with Normal Overlay', 'Q-Q Plot'))
    
    data = df_analysis[selected_var_dist].dropna()
    
    # Histogram with normal overlay
    fig_dist.add_trace(
        go.Histogram(x=data, nbinsx=50, name='Data', opacity=0.7, density=True),
        row=1, col=1
    )
    
    # Normal distribution overlay
    x_norm = np.linspace(data.min(), data.max(), 100)
    y_norm = stats.norm.pdf(x_norm, data.mean(), data.std())
    fig_dist.add_trace(
        go.Scatter(x=x_norm, y=y_norm, mode='lines', name='Normal Distribution', line=dict(color='red')),
        row=1, col=1
    )
    
    # Q-Q plot
    (osm, osr), (slope, intercept, r) = stats.probplot(data, dist="norm", plot=None)
    fig_dist.add_trace(
        go.Scatter(x=osm, y=osr, mode='markers', name='Data Points'),
        row=1, col=2
    )
    fig_dist.add_trace(
        go.Scatter(x=osm, y=slope * osm + intercept, mode='lines', name='Normal Line', line=dict(color='red')),
        row=1, col=2
    )
    
    fig_dist.update_layout(height=500, title=f"Distribution Analysis: {selected_var_dist}")
    st.plotly_chart(fig_dist, use_container_width=True)

# Time Series Decomposition
if "Time Series Decomposition" in analysis_types and hasattr(df_analysis.index, 'freq'):
    st.subheader("🔄 Time Series Decomposition")
    
    decomp_var = st.selectbox("Select variable for decomposition:", selected_vars, key="decomp_var")
    
    # Determine frequency for decomposition
    freq_options = {
        'Daily': 24,
        'Weekly': 24*7,
        'Monthly': 24*30,
        'Auto': None
    }
    
    freq_choice = st.selectbox("Select decomposition frequency:", list(freq_options.keys()))
    freq_value = freq_options[freq_choice]
    
    try:
        # Prepare data for decomposition
        ts_data = df_analysis[decomp_var].dropna()
        
        if freq_value is None:
            # Auto-determine frequency
            freq_value = min(len(ts_data) // 4, 24*7)  # Max one week or quarter of data
        
        if len(ts_data) > 2 * freq_value:
            # Perform decomposition
            decomposition = seasonal_decompose(ts_data, model='additive', period=freq_value)
            
            # Create decomposition plot
            fig_decomp = make_subplots(
                rows=4, cols=1,
                subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'),
                shared_xaxes=True,
                vertical_spacing=0.05
            )
            
            fig_decomp.add_trace(
                go.Scatter(x=ts_data.index, y=ts_data.values, mode='lines', name='Original'),
                row=1, col=1
            )
            
            fig_decomp.add_trace(
                go.Scatter(x=decomposition.trend.index, y=decomposition.trend.values, 
                          mode='lines', name='Trend', line=dict(color='red')),
                row=2, col=1
            )
            
            fig_decomp.add_trace(
                go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal.values, 
                          mode='lines', name='Seasonal', line=dict(color='green')),
                row=3, col=1
            )
            
            fig_decomp.add_trace(
                go.Scatter(x=decomposition.resid.index, y=decomposition.resid.values, 
                          mode='lines', name='Residual', line=dict(color='orange')),
                row=4, col=1
            )
            
            fig_decomp.update_layout(height=800, title=f"Time Series Decomposition: {decomp_var}", showlegend=False)
            st.plotly_chart(fig_decomp, use_container_width=True)
            
            # Decomposition statistics
            st.subheader("📊 Decomposition Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                trend_strength = 1 - np.var(decomposition.resid.dropna()) / np.var(decomposition.trend.dropna() + decomposition.resid.dropna())
                st.metric("Trend Strength", f"{max(0, trend_strength):.3f}")
            
            with col2:
                seasonal_strength = 1 - np.var(decomposition.resid.dropna()) / np.var(decomposition.seasonal.dropna() + decomposition.resid.dropna())
                st.metric("Seasonal Strength", f"{max(0, seasonal_strength):.3f}")
            
            with col3:
                residual_var = np.var(decomposition.resid.dropna())
                st.metric("Residual Variance", f"{residual_var:.3f}")
            
        else:
            st.warning(f"Insufficient data for decomposition. Need at least {2 * freq_value} points, have {len(ts_data)}")
            
    except Exception as e:
        st.error(f"Error in time series decomposition: {str(e)}")

# Stationarity Tests
if "Stationarity Tests" in analysis_types:
    st.subheader("📊 Stationarity Testing")
    
    stationarity_results = []
    
    for var in selected_vars:
        data = df_analysis[var].dropna()
        
        # Augmented Dickey-Fuller test
        adf_stat, adf_p, adf_lags, adf_nobs, adf_critical, adf_icbest = adfuller(data, autolag='AIC')
        adf_stationary = adf_p < 0.05
        
        # KPSS test
        kpss_stat, kpss_p, kpss_lags, kpss_critical = kpss(data, regression='c')
        kpss_stationary = kpss_p > 0.05
        
        stationarity_results.append({
            'Variable': var,
            'ADF Statistic': round(adf_stat, 4),
            'ADF p-value': round(adf_p, 4),
            'ADF Result': 'Stationary' if adf_stationary else 'Non-Stationary',
            'KPSS Statistic': round(kpss_stat, 4),
            'KPSS p-value': round(kpss_p, 4),
            'KPSS Result': 'Stationary' if kpss_stationary else 'Non-Stationary',
            'Overall Assessment': 'Stationary' if (adf_stationary and kpss_stationary) else 'Non-Stationary'
        })
    
    stationarity_df = pd.DataFrame(stationarity_results)
    st.dataframe(stationarity_df, hide_index=True, use_container_width=True)
    
    st.info("""
    **Interpretation:**
    - **ADF Test**: p < 0.05 suggests stationarity (reject null of unit root)
    - **KPSS Test**: p > 0.05 suggests stationarity (fail to reject null of stationarity)
    - Both tests should agree for confident assessment
    """)

# Trend Analysis
if "Trend Analysis" in analysis_types:
    st.subheader("📈 Trend Analysis")
    
    trend_var = st.selectbox("Select variable for trend analysis:", selected_vars, key="trend_var")
    
    data = df_analysis[trend_var].dropna()
    
    # Linear trend analysis
    x_numeric = np.arange(len(data))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, data.values)
    
    # Create trend plot
    fig_trend = go.Figure()
    
    fig_trend.add_trace(go.Scatter(
        x=data.index, y=data.values,
        mode='lines', name='Data',
        line=dict(color='blue', width=1)
    ))
    
    trend_line = slope * x_numeric + intercept
    fig_trend.add_trace(go.Scatter(
        x=data.index, y=trend_line,
        mode='lines', name='Linear Trend',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig_trend.update_layout(
        title=f"Trend Analysis: {trend_var}",
        xaxis_title="Time",
        yaxis_title=trend_var
    )
    
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Trend statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Slope", f"{slope:.6f}")
    
    with col2:
        st.metric("R-squared", f"{r_value**2:.4f}")
    
    with col3:
        st.metric("P-value", f"{p_value:.6f}")
    
    with col4:
        trend_direction = "Increasing" if slope > 0 else "Decreasing" if slope < 0 else "Flat"
        st.metric("Trend Direction", trend_direction)
    
    # Trend significance
    if p_value < 0.05:
        st.success(f"📈 Significant {trend_direction.lower()} trend detected (p < 0.05)")
    else:
        st.info("📊 No significant trend detected (p ≥ 0.05)")

# Statistical Relationships
if "Statistical Relationships" in analysis_types:
    st.subheader("🔗 Statistical Relationships")
    
    if len(selected_vars) >= 2:
        # Variable pair selection
        var1 = st.selectbox("Select first variable:", selected_vars, key="rel_var1")
        var2 = st.selectbox("Select second variable:", [v for v in selected_vars if v != var1], key="rel_var2")
        
        if var1 and var2:
            # Scatter plot with regression
            fig_scatter = px.scatter(
                df_analysis, x=var1, y=var2,
                trendline="ols",
                title=f"Relationship: {var1} vs {var2}"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Statistical tests
            data1 = df_analysis[var1].dropna()
            data2 = df_analysis[var2].dropna()
            
            # Ensure same length
            min_len = min(len(data1), len(data2))
            data1 = data1.iloc[:min_len]
            data2 = data2.iloc[:min_len]
            
            # Correlation tests
            pearson_r, pearson_p = stats.pearsonr(data1, data2)
            spearman_r, spearman_p = stats.spearmanr(data1, data2)
            
            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(data1, data2)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Correlation Analysis")
                st.write(f"**Pearson Correlation:** {pearson_r:.4f} (p = {pearson_p:.6f})")
                st.write(f"**Spearman Correlation:** {spearman_r:.4f} (p = {spearman_p:.6f})")
                
                if pearson_p < 0.05:
                    st.success("Significant linear correlation detected")
                else:
                    st.info("No significant linear correlation")
            
            with col2:
                st.subheader("Linear Regression")
                st.write(f"**Equation:** {var2} = {slope:.4f} × {var1} + {intercept:.4f}")
                st.write(f"**R-squared:** {r_value**2:.4f}")
                st.write(f"**P-value:** {p_value:.6f}")
                st.write(f"**Standard Error:** {std_err:.6f}")

# Statistical Summary
st.header("📋 Statistical Summary")

summary_stats = []
for var in selected_vars:
    data = df_analysis[var].dropna()
    
    # Basic statistics
    stats_dict = {
        'Variable': var,
        'Count': len(data),
        'Mean': data.mean(),
        'Median': data.median(),
        'Std Dev': data.std(),
        'Variance': data.var(),
        'Skewness': stats.skew(data),
        'Kurtosis': stats.kurtosis(data),
        'Min': data.min(),
        'Max': data.max(),
        'Range': data.max() - data.min(),
        'CV': data.std() / data.mean() if data.mean() != 0 else 0
    }
    
    summary_stats.append(stats_dict)

summary_df = pd.DataFrame(summary_stats)
for col in summary_df.columns:
    if col != 'Variable' and col != 'Count':
        summary_df[col] = summary_df[col].round(4)

st.dataframe(summary_df, hide_index=True, use_container_width=True)

# Key Statistical Insights
st.header("💡 Key Statistical Insights")

insights = []

# Correlation insights
if "Correlation Analysis" in analysis_types and len(selected_vars) >= 2:
    max_corr = 0
    max_pair = None
    for i in range(len(selected_vars)):
        for j in range(i + 1, len(selected_vars)):
            corr_val = abs(pearson_corr.iloc[i, j])
            if corr_val > max_corr:
                max_corr = corr_val
                max_pair = (selected_vars[i], selected_vars[j])
    
    if max_pair:
        insights.append(f"🔗 **Strongest Correlation**: {max_pair[0]} and {max_pair[1]} (r = {max_corr:.3f})")

# Variability insights
cv_values = [(var, summary_df[summary_df['Variable'] == var]['CV'].iloc[0]) for var in selected_vars]
most_variable = max(cv_values, key=lambda x: x[1])
least_variable = min(cv_values, key=lambda x: x[1])

insights.append(f"📊 **Most Variable**: {most_variable[0]} (CV = {most_variable[1]:.3f})")
insights.append(f"📊 **Least Variable**: {least_variable[0]} (CV = {least_variable[1]:.3f})")

# Normality insights
if "Normality Tests" in analysis_types:
    normal_vars = [result['Variable'] for result in normality_results if result['Overall Assessment'] == 'Normal']
    if normal_vars:
        insights.append(f"📈 **Normal Distributions**: {', '.join(normal_vars)}")
    else:
        insights.append("📈 **No variables follow normal distribution** - consider data transformation")

# Display insights
for insight in insights:
    st.info(insight)

# Export section
st.header("📥 Export Statistical Results")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Export Summary Statistics"):
        summary_csv = summary_df.to_csv(index=False)
        st.download_button(
            label="Download Summary CSV",
            data=summary_csv,
            file_name="statistical_summary.csv",
            mime="text/csv"
        )

with col2:
    if st.button("🔗 Export Correlation Matrix"):
        if "Correlation Analysis" in analysis_types:
            corr_csv = pearson_corr.to_csv()
            st.download_button(
                label="Download Correlation CSV",
                data=corr_csv,
                file_name="correlation_matrix.csv",
                mime="text/csv"
            )

with col3:
    if st.button("📋 Export All Results"):
        # Combine all results
        all_results = {
            'summary_statistics': summary_df.to_dict('records'),
            'selected_variables': selected_vars,
            'analysis_types': analysis_types,
            'sample_info': {
                'total_records': len(df_analysis),
                'date_range': f"{df_analysis.index.min()} to {df_analysis.index.max()}"
            }
        }
        
        if "Correlation Analysis" in analysis_types:
            all_results['correlation_matrix'] = pearson_corr.to_dict()
        
        if "Normality Tests" in analysis_types:
            all_results['normality_tests'] = normality_results
        
        if "Stationarity Tests" in analysis_types:
            all_results['stationarity_tests'] = stationarity_df.to_dict('records')
        
        import json
        results_json = json.dumps(all_results, indent=2, default=str)
        st.download_button(
            label="Download All Results JSON",
            data=results_json,
            file_name="statistical_analysis_results.json",
            mime="application/json"
        )
