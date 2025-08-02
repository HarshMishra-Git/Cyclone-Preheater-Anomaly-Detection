import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import json
import base64
from io import StringIO

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.data_processor import DataProcessor
from utils.anomaly_detector import AnomalyDetector
from utils.visualizer import Visualizer
from utils.report_generator import ReportGenerator

st.set_page_config(
    page_title="Report Generator - Cyclone Analysis",
    page_icon="📋",
    layout="wide"
)

st.title("📋 Automated Report Generator")

# Check if data is loaded
if not st.session_state.get('data_loaded', False) or st.session_state.get('processed_data') is None:
    st.warning("⚠️ Please load data first from the main page.")
    st.stop()

df = st.session_state.processed_data
report_gen = ReportGenerator()
processor = DataProcessor()
detector = AnomalyDetector()

# Initialize session state for report
if 'report_data' not in st.session_state:
    st.session_state.report_data = None
if 'report_generated' not in st.session_state:
    st.session_state.report_generated = False

# Sidebar controls
st.sidebar.header("📊 Report Configuration")

# Report sections selection
report_sections = st.sidebar.multiselect(
    "Select Report Sections",
    [
        "Executive Summary",
        "Data Quality Assessment", 
        "Anomaly Detection Results",
        "Statistical Analysis",
        "Key Insights",
        "Recommendations"
    ],
    default=[
        "Executive Summary",
        "Data Quality Assessment",
        "Anomaly Detection Results",
        "Key Insights",
        "Recommendations"
    ]
)

# Anomaly detection parameters for report
st.sidebar.subheader("🔧 Analysis Parameters")
enable_anomaly_detection = st.sidebar.checkbox("Run Anomaly Detection for Report", value=True)

if enable_anomaly_detection:
    report_methods = st.sidebar.multiselect(
        "Anomaly Detection Methods",
        ["zscore", "iqr", "isolation_forest", "statistical"],
        default=["zscore", "iqr", "isolation_forest"]
    )
    
    contamination_level = st.sidebar.slider("Contamination Level", 0.01, 0.2, 0.1, 0.01)
    zscore_threshold = st.sidebar.slider("Z-Score Threshold", 2.0, 4.0, 3.0, 0.1)

# Variable selection
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
selected_variables = st.sidebar.multiselect(
    "Variables for Analysis",
    numeric_cols,
    default=numeric_cols
)

# Report format options
st.sidebar.subheader("📄 Export Options")
export_formats = st.sidebar.multiselect(
    "Export Formats",
    ["HTML Report", "JSON Data", "CSV Summary", "PowerPoint Content"],
    default=["HTML Report", "JSON Data"]
)

# Generate report button
generate_report = st.sidebar.button("🚀 Generate Report", type="primary")

# Main content
if generate_report and selected_variables:
    
    with st.spinner("Generating comprehensive report..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Data quality analysis
        status_text.text("Analyzing data quality...")
        progress_bar.progress(0.2)
        
        # Step 2: Run anomaly detection if enabled
        anomaly_summary = {}
        anomaly_periods = []
        
        if enable_anomaly_detection and report_methods:
            status_text.text("Running anomaly detection...")
            progress_bar.progress(0.4)
            
            # Run selected anomaly detection methods
            results_df = df.copy()
            
            if 'zscore' in report_methods:
                results_df = detector.detect_zscore_anomalies(results_df, zscore_threshold, selected_variables)
            
            if 'iqr' in report_methods:
                results_df = detector.detect_iqr_anomalies(results_df, 1.5, selected_variables)
            
            if 'isolation_forest' in report_methods:
                results_df = detector.detect_isolation_forest_anomalies(
                    results_df, contamination=contamination_level, columns=selected_variables
                )
            
            if 'statistical' in report_methods:
                results_df = detector.detect_statistical_anomalies(results_df, 24, 2.0, selected_variables)
            
            # Combine methods
            results_df = detector.combine_anomaly_methods(results_df, report_methods)
            
            # Generate summaries
            anomaly_summary = detector.get_anomaly_summary(results_df)
            anomaly_periods = detector.get_anomaly_periods(results_df, 'combined', min_duration_hours=1)
        
        # Step 3: Generate report data
        status_text.text("Compiling report data...")
        progress_bar.progress(0.7)
        
        report_data = report_gen.generate_report_data(df, anomaly_summary, anomaly_periods)
        
        # Step 4: Store results
        status_text.text("Finalizing report...")
        progress_bar.progress(1.0)
        
        st.session_state.report_data = report_data
        st.session_state.report_generated = True
        
        status_text.text("Report generated successfully! ✅")
        
        # Auto-refresh to show results
        st.rerun()

# Display report if generated
if st.session_state.report_generated and st.session_state.report_data:
    report_data = st.session_state.report_data
    
    # Report header
    st.header("📊 Cyclone System Analysis Report")
    
    # Executive Summary section
    if "Executive Summary" in report_sections:
        st.subheader("📋 Executive Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_records = report_data['metadata']['data_quality']['total_records']
            st.metric("Total Records", f"{total_records:,}")
        
        with col2:
            duration_days = report_data['metadata']['data_quality']['date_range']['duration_days']
            st.metric("Analysis Period", f"{duration_days} days")
        
        with col3:
            total_anomalies = report_data['key_insights']['total_anomalies']
            st.metric("Anomalies Detected", f"{total_anomalies:,}")
        
        with col4:
            anomaly_rate = report_data['key_insights']['anomaly_rate_percent']
            st.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")
        
        # System health indicator
        st.subheader("🏥 System Health Assessment")
        
        if anomaly_rate < 1:
            st.success("🟢 **EXCELLENT** - System operating within optimal parameters")
        elif anomaly_rate < 5:
            st.info("🟡 **GOOD** - System performance is acceptable with minor deviations")
        elif anomaly_rate < 10:
            st.warning("🟠 **MODERATE** - System shows concerning patterns requiring attention")
        else:
            st.error("🔴 **CRITICAL** - System requires immediate investigation and maintenance")
    
    # Data Quality Assessment
    if "Data Quality Assessment" in report_sections:
        st.subheader("📊 Data Quality Assessment")
        
        quality_data = report_data['metadata']['data_quality']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Data Completeness**")
            missing_pct = quality_data['missing_data_percent']
            if missing_pct < 1:
                st.success(f"✅ Excellent data quality ({100-missing_pct:.1f}% complete)")
            elif missing_pct < 5:
                st.info(f"ℹ️ Good data quality ({100-missing_pct:.1f}% complete)")
            else:
                st.warning(f"⚠️ Data quality concerns ({100-missing_pct:.1f}% complete)")
        
        with col2:
            st.write("**Analysis Coverage**")
            start_date = quality_data['date_range']['start']
            end_date = quality_data['date_range']['end']
            st.write(f"**Period:** {start_date} to {end_date}")
            st.write(f"**Variables:** {len(report_data['data_preparation']['variables'])} sensor measurements")
        
        # Data preprocessing summary
        st.write("**Data Preprocessing Applied:**")
        for step in report_data['analysis_strategy']['data_preprocessing']:
            st.write(f"• {step}")
    
    # Anomaly Detection Results
    if "Anomaly Detection Results" in report_sections and enable_anomaly_detection:
        st.subheader("⚠️ Anomaly Detection Results")
        
        # Method comparison table
        if report_data['key_insights']['method_comparison']:
            method_df = pd.DataFrame([
                {
                    'Detection Method': method.title().replace('_', ' '),
                    'Anomalies Found': stats['count'],
                    'Detection Rate': f"{stats['rate_percent']:.2f}%",
                    'First Detection': stats['first_occurrence'] or 'None',
                    'Last Detection': stats['last_occurrence'] or 'None'
                }
                for method, stats in report_data['key_insights']['method_comparison'].items()
            ])
            
            st.dataframe(method_df, hide_index=True, use_container_width=True)
        
        # Critical periods
        critical_periods = report_data['key_insights']['critical_periods']
        if critical_periods:
            st.subheader("🚨 Critical Anomaly Periods")
            
            for i, period in enumerate(critical_periods[:5]):  # Show top 5
                with st.expander(f"Period {i+1}: {period['start_time']} - {period['end_time']}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Duration", f"{period['duration_hours']:.1f} hours")
                    
                    with col2:
                        st.metric("Anomaly Count", period['anomaly_count'])
                    
                    with col3:
                        severity = period.get('severity_score', 0)
                        st.metric("Severity Score", f"{severity:.3f}")
        
        else:
            st.info("No significant anomaly periods detected.")
    
    # Statistical Analysis
    if "Statistical Analysis" in report_sections:
        st.subheader("📈 Statistical Analysis")
        
        # Variable statistics
        stats_data = report_data['data_preparation']['statistics']
        
        if stats_data:
            stats_df = pd.DataFrame([
                {
                    'Variable': var,
                    'Mean': f"{data['mean']:.2f}",
                    'Std Dev': f"{data['std']:.2f}",
                    'Min': f"{data['min']:.2f}",
                    'Max': f"{data['max']:.2f}",
                    'Coefficient of Variation': f"{(data['std']/data['mean'] if data['mean'] != 0 else 0):.3f}"
                }
                for var, data in stats_data.items()
            ])
            
            st.dataframe(stats_df, hide_index=True, use_container_width=True)
        
        # Correlation insights
        correlations = report_data['data_preparation']['correlations']
        if correlations:
            st.subheader("🔗 Strong Variable Relationships")
            
            corr_df = pd.DataFrame(correlations)
            corr_df['correlation'] = corr_df['correlation'].round(3)
            corr_df.columns = ['Variable 1', 'Variable 2', 'Correlation']
            
            st.dataframe(corr_df, hide_index=True, use_container_width=True)
    
    # Key Insights
    if "Key Insights" in report_sections:
        st.subheader("💡 Key Insights")
        
        insights = report_data['key_insights']['insights']
        
        for i, insight in enumerate(insights, 1):
            st.info(f"**Insight {i}:** {insight}")
    
    # Recommendations
    if "Recommendations" in report_sections:
        st.subheader("🎯 Operational Recommendations")
        
        recommendations = report_data['recommendations']
        
        # Categorize recommendations
        immediate_actions = []
        maintenance_actions = []
        monitoring_actions = []
        
        for rec in recommendations:
            if any(word in rec.lower() for word in ['immediate', 'critical', 'urgent']):
                immediate_actions.append(rec)
            elif any(word in rec.lower() for word in ['maintenance', 'calibration', 'preventive']):
                maintenance_actions.append(rec)
            else:
                monitoring_actions.append(rec)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if immediate_actions:
                st.write("**🚨 Immediate Actions**")
                for action in immediate_actions:
                    st.error(f"• {action}")
        
        with col2:
            if maintenance_actions:
                st.write("**🔧 Maintenance & Calibration**")
                for action in maintenance_actions:
                    st.warning(f"• {action}")
        
        with col3:
            if monitoring_actions:
                st.write("**👁️ Monitoring & Prevention**")
                for action in monitoring_actions:
                    st.info(f"• {action}")
    
    # Export section
    st.header("📥 Export Report")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # HTML Report
    if "HTML Report" in export_formats:
        with col1:
            if st.button("📄 Generate HTML Report"):
                html_content = report_gen.generate_html_report()
                
                st.download_button(
                    label="Download HTML Report",
                    data=html_content,
                    file_name=f"cyclone_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
                    mime="text/html"
                )
    
    # JSON Data
    if "JSON Data" in export_formats:
        with col2:
            if st.button("📊 Export JSON Data"):
                json_content = report_gen.export_to_json()
                
                st.download_button(
                    label="Download JSON Report",
                    data=json_content,
                    file_name=f"cyclone_analysis_data_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json"
                )
    
    # CSV Summary
    if "CSV Summary" in export_formats:
        with col3:
            if st.button("📈 Export CSV Summary"):
                summary_csv, periods_csv = report_gen.export_to_csv(df, report_data['key_insights']['critical_periods'])
                
                # Create a combined CSV with both summary and periods
                combined_csv = "=== ANOMALY DETECTION SUMMARY ===\n" + summary_csv + "\n\n=== ANOMALY PERIODS ===\n" + periods_csv
                
                st.download_button(
                    label="Download CSV Summary",
                    data=combined_csv,
                    file_name=f"cyclone_analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
    
    # PowerPoint Content
    if "PowerPoint Content" in export_formats:
        with col4:
            if st.button("📋 Export PowerPoint Content"):
                ppt_content = report_gen.create_powerpoint_content()
                
                # Convert to formatted text for download
                ppt_text = "CYCLONE ANALYSIS - POWERPOINT CONTENT\n" + "="*50 + "\n\n"
                
                for slide_id, slide_data in ppt_content.items():
                    ppt_text += f"SLIDE: {slide_data['title']}\n"
                    ppt_text += "-" * 40 + "\n"
                    
                    if 'subtitle' in slide_data:
                        ppt_text += f"Subtitle: {slide_data['subtitle']}\n\n"
                    
                    ppt_text += "Content:\n"
                    for content_item in slide_data['content']:
                        ppt_text += f"• {content_item}\n"
                    
                    ppt_text += "\n" + "="*50 + "\n\n"
                
                st.download_button(
                    label="Download PowerPoint Content",
                    data=ppt_text,
                    file_name=f"cyclone_analysis_presentation_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain"
                )
    
    # Report preview section
    st.header("👁️ Report Preview")
    
    preview_format = st.selectbox(
        "Select preview format:",
        ["HTML Preview", "PowerPoint Slides", "JSON Structure"],
        key="preview_format"
    )
    
    if preview_format == "HTML Preview":
        html_content = report_gen.generate_html_report()
        st.components.v1.html(html_content, height=800, scrolling=True)
    
    elif preview_format == "PowerPoint Slides":
        ppt_content = report_gen.create_powerpoint_content()
        
        for slide_id, slide_data in ppt_content.items():
            with st.expander(f"📄 {slide_data['title']}"):
                if 'subtitle' in slide_data:
                    st.write(f"**{slide_data['subtitle']}**")
                    st.write("")
                
                for content_item in slide_data['content']:
                    st.write(content_item)
    
    elif preview_format == "JSON Structure":
        st.json(report_data)

else:
    # Initial instructions
    st.info("""
    ## 📋 Automated Report Generator
    
    Generate comprehensive analysis reports for your cyclone system data with just a few clicks.
    
    ### 🚀 Getting Started:
    
    1. **Configure Report Sections**: Select which sections to include in your report
    2. **Set Analysis Parameters**: Choose anomaly detection methods and thresholds
    3. **Select Variables**: Pick which sensor variables to analyze
    4. **Choose Export Formats**: Select your preferred output formats
    5. **Generate Report**: Click the generate button to create your report
    
    ### 📊 Available Report Sections:
    
    - **Executive Summary**: High-level overview with key metrics
    - **Data Quality Assessment**: Completeness and reliability analysis
    - **Anomaly Detection Results**: Detailed anomaly findings and patterns
    - **Statistical Analysis**: Descriptive statistics and relationships
    - **Key Insights**: Automated insights and pattern recognition
    - **Recommendations**: Actionable operational recommendations
    
    ### 📄 Export Formats:
    
    - **HTML Report**: Professional web-based report for viewing and sharing
    - **JSON Data**: Machine-readable data for further analysis
    - **CSV Summary**: Tabular data for spreadsheet applications
    - **PowerPoint Content**: Structured content for presentation creation
    
    ### 💡 Tips:
    
    - Include anomaly detection for the most comprehensive analysis
    - Select all variables for complete system assessment
    - Use HTML format for professional presentation
    - Export JSON for integration with other systems
    """)
    
    # Show current data summary
    st.subheader("📊 Current Dataset Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    
    with col2:
        st.metric("Variables Available", len(numeric_cols))
    
    with col3:
        if hasattr(df.index, 'min'):
            duration = (df.index.max() - df.index.min()).days
            st.metric("Data Duration", f"{duration} days")
        else:
            st.metric("Data Duration", "N/A")
    
    with col4:
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("Data Completeness", f"{100-missing_pct:.1f}%")
