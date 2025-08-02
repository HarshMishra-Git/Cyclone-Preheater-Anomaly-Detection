import streamlit as st
import pandas as pd
import numpy as np
from utils.data_processor import DataProcessor
from utils.anomaly_detector import AnomalyDetector
from utils.visualizer import Visualizer
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Cyclone Anomaly Detection",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

def main():
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("🌪️ Cyclone Anomaly Detection System")
    st.markdown("**Performance-Optimized Industrial Analytics Dashboard**")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    st.sidebar.markdown("---")
    
    # File upload
    st.sidebar.header("Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Choose CSV file", 
        type=['csv'],
        help="Upload your cyclone sensor data CSV file"
    )
    
    # Check if file is uploaded or if we have the default file
    data_file = uploaded_file
    if not data_file:
        # local data directory
        data_paths = [
            "data/data.csv",  # Local path
            "data.csv"  # Root directory
        ]
        
        for path in data_paths:
            try:
                import os
                if os.path.exists(path):
                    data_file = path
                    st.sidebar.success(f"Using dataset: {os.path.basename(path)}")
                    break
            except:
                continue
        
        if not data_file:
            st.sidebar.warning("Please upload a CSV file or data.csv' in the project directory")
            return
    
    # Load and process data
    if data_file and not st.session_state.data_loaded:
        with st.spinner("Loading and processing data..."):
            try:
                processor = DataProcessor()
                if isinstance(data_file, str):
                    # File path
                    df = processor.load_data(data_file)
                else:
                    # Uploaded file
                    df = processor.load_data(data_file)
                
                st.session_state.processed_data = df
                st.session_state.data_loaded = True
                st.sidebar.success(f"✅ Loaded {len(df):,} records")
                
            except Exception as e:
                st.sidebar.error(f"Error loading data: {str(e)}")
                return
    
    # Main content
    if st.session_state.data_loaded and st.session_state.processed_data is not None:
        df = st.session_state.processed_data
        
        # Overview metrics
        st.header("📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        
        with col2:
            st.metric("Date Range", f"{(df.index[-1] - df.index[0]).days} days")
        
        with col3:
            st.metric("Variables", len(df.columns))
        
        with col4:
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.metric("Missing Data", f"{missing_pct:.2f}%")
        
        st.markdown("---")
        
        # Quick visualization
        st.header("🎯 Quick Analysis")
        
        # Select variable for quick view
        selected_var = st.selectbox(
            "Select variable for quick visualization:",
            df.columns.tolist(),
            index=0
        )
        
        # Create quick plot
        visualizer = Visualizer()
        fig = visualizer.create_time_series_plot(df, selected_var)
        st.plotly_chart(fig, use_container_width=True)
        
        # Basic statistics
        st.subheader("📈 Basic Statistics")
        st.dataframe(df.describe(), use_container_width=True)
        
        # Navigation info
        st.markdown("---")
        st.info("📱 Use the sidebar to navigate to different analysis pages:")
        st.markdown("""
        - **Data Overview**: Detailed data exploration and quality assessment
        - **Anomaly Detection**: Run various anomaly detection algorithms
        - **Statistical Analysis**: Correlation analysis and advanced statistics
        - **Report Generator**: Generate automated PowerPoint reports
        """)
        
    else:
        # Landing page
        st.markdown("""
        ## Welcome to Cyclone Anomaly Detection System
        
        This application helps you detect anomalies in cyclone preheater systems using advanced statistical methods and machine learning algorithms.
        
        ### Features:
        - 🚀 **Performance Optimized**: Handles large datasets (370K+ records) efficiently
        - 📊 **Multiple Detection Methods**: Z-Score, IQR, Isolation Forest, and more
        - 📈 **Interactive Visualizations**: Real-time charts with anomaly highlights
        - 📋 **Automated Reports**: Generate PowerPoint presentations automatically
        - 🔍 **Statistical Analysis**: Correlation studies and trend analysis
        
        ### Getting Started:
        1. Upload your cyclone sensor data CSV file using the sidebar
        2. Navigate through different analysis pages using the menu
        3. Configure detection parameters and run analysis
        4. Export results and generate reports
        
        ### Expected Data Format:
        - **Time column**: Timestamp data (will be parsed automatically)
        - **Sensor variables**: Temperature and pressure readings
        - **Frequency**: Regular time intervals (e.g., every 5 minutes)
        """)

if __name__ == "__main__":
    main()
