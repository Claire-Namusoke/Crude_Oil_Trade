# -*- coding: utf-8 -*-
import pyodbc
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sqlalchemy import create_engine
import urllib.parse
import os
import re
import subprocess

# OpenAI imports
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    st.warning("⚠️ OpenAI library not installed. AI features will use fallback analysis.")

# Set up OpenAI client if available
if OPENAI_AVAILABLE:
    # Try Streamlit secrets first, then fall back to environment variable
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except:
        api_key = os.getenv('OPENAI_API_KEY')
    
    if api_key:
        client = OpenAI(api_key=api_key)
        AI_ENABLED = True
    else:
        client = None
        AI_ENABLED = False
else:
    client = None
    AI_ENABLED = False

# --- Page Configuration ---
st.set_page_config(
    page_title="🌍 Global Crude Oil Trade Dashboard (1995-2021)",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Deep Black Theme ---
st.markdown("""
<style>
    /* Root and HTML background - deepest black */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    
    /* Main app background - match chart background exactly */
    .main {
        background-color: #000000 !important;
    }
    
    /* Main content container */
    .main .block-container {
        background-color: #000000 !important;
        color: #ffffff !important;
        padding: 1rem 1.5rem;
        max-width: 100%;
    }
    
    /* Sidebar - deep black theme */
    .sidebar .sidebar-content {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    
    .sidebar {
        background-color: #000000 !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: #000000 !important;
    }
    
    [data-testid="stSidebar"] > div {
        background-color: #000000 !important;
    }
    
    /* Headers - bright white */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        background-color: transparent !important;
    }
    
    /* Main title - single line */
    h1 {
        white-space: nowrap !important;
        overflow: visible !important;
        font-size: 1.6rem !important;
        margin: 0.3rem 0 !important;
    }
    
    /* Responsive title for smaller screens */
    @media (max-width: 1200px) {
        h1 {
            font-size: 1.6rem !important;
        }
    }
    
    @media (max-width: 992px) {
        h1 {
            font-size: 1.4rem !important;
        }
    }
    
    @media (max-width: 768px) {
        h1 {
            font-size: 1.2rem !important;
        }
    }
    
    @media (max-width: 576px) {
        h1 {
            font-size: 1rem !important;
        }
    }
    
    /* Text elements */
    .stMarkdown, .stText, p, span, div {
        color: #ffffff !important;
        background-color: transparent !important;
    }
    
    /* Metrics - pure black background */
    [data-testid="metric-container"] {
        background-color: #000000 !important;
        border: 1px solid #333333 !important;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(255,255,255,0.1);
    }
    
    [data-testid="metric-container"] > div {
        color: #ffffff !important;
        background-color: transparent !important;
    }
    
    [data-testid="metric-container"] label {
        color: #cccccc !important;
    }
    
    /* Custom KPI Cards with Cyan Values */
    .kpi-card {
        background-color: #000000 !important;
        border: 1px solid #333333 !important;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,255,255,0.2);
        transition: all 0.3s ease;
    }
    
    .kpi-card:hover {
        border-color: #00FFFF !important;
        box-shadow: 0 4px 16px rgba(0,255,255,0.3);
    }
    
    .kpi-title {
        color: #ffffff !important;
        font-size: 1.2rem !important;
        font-weight: bold !important;
        margin-bottom: 10px !important;
    }
    
    .kpi-value {
        color: #00FFFF !important;
        font-size: 2.5rem !important;
        font-weight: bold !important;
        margin: 0 !important;
        text-shadow: 0 0 10px rgba(0,255,255,0.5);
    }
    
    .kpi-subvalue {
        color: #00FFFF !important;
        font-size: 1.5rem !important;
        font-weight: bold !important;
        margin: 5px 0 0 0 !important;
        text-shadow: 0 0 8px rgba(0,255,255,0.4);
    }
    
    /* Ensure all h2 elements in KPI sections are cyan */
    div h2 {
        color: #00FFFF !important;
        text-shadow: 0 0 10px rgba(0,255,255,0.6) !important;
    }
    
    /* Strong cyan override for any text elements */
    [data-testid="column"] h2 {
        color: #00FFFF !important;
        text-shadow: 0 0 12px rgba(0,255,255,0.7) !important;
        font-weight: bold !important;
    }
    
    /* Force cyan color for action analysis boxes */
    .action-value {
        color: #00FFFF !important;
        text-shadow: 0 0 8px rgba(0,255,255,0.4) !important;
    }
    
    /* AI Response styling - Enhanced cyan with glow effect */
    .ai-response {
        color: #00FFFF !important;
        font-size: 16px !important;
        line-height: 1.6 !important;
        text-shadow: 0 0 10px rgba(0,255,255,0.5) !important;
        background: linear-gradient(135deg, rgba(0,255,255,0.05), rgba(0,255,255,0.02)) !important;
        border-left: 3px solid #00FFFF !important;
        padding: 15px !important;
        border-radius: 8px !important;
        margin: 10px 0 !important;
        font-weight: 500 !important;
    }
    
    /* Enhanced styling for AI response text */
    .ai-response p, .ai-response ul, .ai-response li, .ai-response strong {
        color: #00FFFF !important;
    }
    
    /* Ensure all markdown elements in AI response are cyan */
    .ai-response * {
        color: #00FFFF !important;
    }
    
    /* Override any conflicting styles for AI responses */
    div[style*="color: #00FFFF"] {
        color: #00FFFF !important;
        text-shadow: 0 0 8px rgba(0,255,255,0.4) !important;
    }
    
    /* Force cyan color for ALL YoY percentage and numerical values */
    span[style*="color:#00FFFF"] {
        color: #00FFFF !important;
    }
    
    div[style*="color:#00FFFF"] {
        color: #00FFFF !important;
    }
    
    /* Target specific YoY metrics containers */
    div[style*="flex:1"][style*="border:1px solid #333333"] span,
    div[style*="flex:1"][style*="border:1px solid #333333"] div {
        color: #00FFFF !important;
    }
    
    /* Override any markdown span elements that should be cyan */
    .element-container span[style*="#00FFFF"],
    .stMarkdown span[style*="#00FFFF"],
    .stMarkdown div[style*="#00FFFF"] {
        color: #00FFFF !important;
        font-weight: bold !important;
        text-shadow: 0 0 8px rgba(0,255,255,0.4) !important;
    }
    
    /* Aggressive override for any numerical values in YoY section */
    div[style*="padding:10px"][style*="border-radius:8px"] div[style*="font-weight:700"] {
        color: #00FFFF !important;
    }
    
    /* YoY Metrics CSS Classes */
    .yoy-metric-box {
        flex: 1;
        min-width: 140px;
        padding: 10px;
        border: 1px solid #333333;
        border-radius: 8px;
        background: #000000;
    }
    
    .yoy-metric-label {
        color: #ffffff !important;
        font-size: 0.85rem;
        margin-bottom: 6px;
    }
    
    .yoy-metric-value {
        color: #00FFFF !important;
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        text-shadow: 0 0 8px rgba(0,255,255,0.35) !important;
    }
    
    .yoy-metric-large {
        color: #00FFFF !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        text-shadow: 0 0 8px rgba(0,255,255,0.35) !important;
    }
    
    .yoy-trend-box {
        padding: 8px;
        border-radius: 6px;
        background: #000000;
    }
    
    .yoy-trend-value {
        color: #00FFFF !important;
        font-weight: 700 !important;
        margin-left: 6px !important;
    }
    
    /* Specific color classes for YoY percentages */
    .yoy-positive {
        color: #00FFFF !important;
        font-weight: 700 !important;
        text-shadow: 0 0 8px rgba(0,255,255,0.35) !important;
    }
    
    .yoy-negative {
        color: #FF4444 !important;
        font-weight: 700 !important;
        text-shadow: 0 0 8px rgba(255,68,68,0.35) !important;
    }
    
    .yoy-year {
        color: #ffffff !important;
        font-weight: 700 !important;
        text-shadow: 0 0 5px rgba(255,255,255,0.3) !important;
    }
    
    /* Input elements - match chart background */
    .stSelectbox > div > div {
        background-color: #000000 !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
    }
    
    .stMultiSelect > div > div {
        background-color: #000000 !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
    }
    
    .stTextInput > div > div > input {
        background-color: #000000 !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
    }
    
    /* Dropdown options */
    .stSelectbox [data-baseweb="select"] {
        background-color: #000000 !important;
    }
    
    /* Tabs - pure black */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #000000 !important;
        border-bottom: 1px solid #333333;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #000000 !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #222222 !important;
        border-bottom: 2px solid #ffffff !important;
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        background-color: #000000 !important;
    }
    
    /* Dataframes - black background */
    .dataframe, [data-testid="dataframe"] {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    
    .dataframe tbody tr {
        background-color: #000000 !important;
    }
    
    .dataframe tbody tr:nth-child(even) {
        background-color: #111111 !important;
    }
    
    .dataframe thead th {
        background-color: #222222 !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
    }
    
    .dataframe tbody td {
        background-color: transparent !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
    }
    
    /* Buttons - black theme */
    .stDownloadButton button {
        background-color: #000000 !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
    }
    
    .stDownloadButton button:hover {
        background-color: #222222 !important;
        border-color: #ffffff !important;
    }
    
    /* Alert boxes - black background */
    .stSuccess {
        background-color: #000000 !important;
        border-left: 4px solid #00ff00 !important;
        color: #ffffff !important;
    }
    
    .stInfo {
        background-color: #000000 !important;
        border-left: 4px solid #00bfff !important;
        color: #ffffff !important;
    }
    
    .stWarning {
        background-color: #000000 !important;
        border-left: 4px solid #ffaa00 !important;
        color: #ffffff !important;
    }
    
    .stError {
        background-color: #000000 !important;
        border-left: 4px solid #ff0000 !important;
        color: #ffffff !important;
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background-color: #00ff00 !important;
    }
    
    .stProgress > div {
        background-color: #333333 !important;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #000000 !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
    }
    
    /* Separators */
    hr {
        border-color: #333333 !important;
        background-color: #333333 !important;
    }
    
    /* Column containers */
    [data-testid="column"] {
        background-color: transparent !important;
        padding: 0.25rem !important;
    }
    
    /* Remove any remaining white backgrounds */
    .element-container {
        background-color: transparent !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Compact spacing for sections */
    .stMarkdown {
        margin-bottom: 0.5rem !important;
    }
    
    /* Reduce space around charts */
    .js-plotly-plot .plotly {
        margin: 0 !important;
    }
    
    /* Compact subheaders */
    h2, h3 {
        margin: 0.3rem 0 0.2rem 0 !important;
        font-size: 1.1rem !important;
    }
    
    /* Reduce gap between columns */
    .row-widget.stHorizontal > div {
        gap: 0.3rem !important;
    }
    
    /* More compact spacing for all sections */
    .element-container {
        margin-bottom: 0.3rem !important;
    }
    
    /* Tighter column padding */
    [data-testid="column"] {
        padding: 0.15rem !important;
    }
    
    .stApp {
        background-color: #000000 !important;
    }
    
    /* Labels */
    label {
        color: #ffffff !important;
    }
    
    /* Ensure plot containers match */
    .js-plotly-plot {
        background-color: #000000 !important;
    }
    
    /* Streamlit native elements */
    [data-testid="stHeader"] {
        background-color: transparent !important;
    }
    
    [data-testid="stToolbar"] {
        background-color: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Data Loading Function ---
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_crude_oil_data():
    """Load crude oil data from CSV (cloud-ready) or database (local fallback)"""
    
    # Try CSV file first (for cloud deployment and reliability)
    csv_path = "crude_oil_data.csv"
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            st.success(f"✅ Loaded {len(df)} records from CSV file")
            return df, None
        except Exception as csv_error:
            st.warning(f"CSV file exists but failed to load: {csv_error}")
    
    # Fallback to database connection for local development
    database = 'CrudeOilTrade'
    working_connection_string = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER=CLAIRE-NAMUSOKE\\SQLEXPRESS;"
        f"DATABASE={database};"
        f"Trusted_Connection=yes;"
        f"Encrypt=yes;"
        f"TrustServerCertificate=yes;"
        f"Connection Timeout=60;"
        f"Login Timeout=60;"
    )
    
    try:
        import warnings
        warnings.filterwarnings('ignore', message='.*pandas only supports SQLAlchemy.*')
        
        conn = pyodbc.connect(working_connection_string)
        query = "SELECT * FROM CrudeOilData;"
        df = pd.read_sql(query, conn)
        conn.close()
        
        st.info(f"📊 Loaded {len(df)} records from database")
        return df, None
        
    except Exception as direct_error:
        servers_to_try = [
            r'CLAIRE-NAMUSOKE\SQLEXPRESS',
            r'localhost\SQLEXPRESS',
            r'.\SQLEXPRESS',
            r'(local)\SQLEXPRESS'
        ]
    
    # Try different server formats
    for server in servers_to_try:
        try:
            # Create SQLAlchemy connection string with proper URL encoding
            params = urllib.parse.quote_plus(
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={server};"
                f"DATABASE={database};"
                f"Trusted_Connection=yes;"
                f"Encrypt=yes;"
                f"TrustServerCertificate=yes;"
            )
            connection_string = f"mssql+pyodbc:///?odbc_connect={params}"
            
            # Create SQLAlchemy engine with additional timeout settings
            engine = create_engine(
                connection_string,
                pool_timeout=60,
                pool_recycle=-1,
                connect_args={
                    "timeout": 60,
                    "login_timeout": 60,
                    "connection_timeout": 60
                }
            )
            
            # Test connection first with a simple query
            test_query = "SELECT 1 as test"
            pd.read_sql(test_query, engine)
            
            # Use the engine with pandas for actual data
            query = "SELECT * FROM CrudeOilData;"
            df = pd.read_sql(query, engine)
            
            # Dispose of the engine to free resources
            engine.dispose()
            
            return df, None
            
        except Exception as e:
            continue
    
    # If all SQLAlchemy attempts fail, try direct pyodbc
    for server in servers_to_try:
        try:
            import warnings
            warnings.filterwarnings('ignore', message='.*SQLAlchemy.*')
            
            # Direct pyodbc connection as fallback
            pyodbc_conn_string = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={server};"
                f"DATABASE={database};"
                f"Trusted_Connection=yes;"
                f"Encrypt=yes;"
                f"TrustServerCertificate=yes;"
                f"Connection Timeout=60;"
                f"Login Timeout=60;"
            )
            
            conn = pyodbc.connect(pyodbc_conn_string)
            query = "SELECT * FROM CrudeOilData;"
            df = pd.read_sql(query, conn)
            conn.close()
            
            return df, None
            
        except Exception as e:
            continue
    
    # If all attempts fail
    error_msg = f"❌ All connection attempts failed for all server formats: {', '.join(servers_to_try)}"
    return pd.DataFrame(), error_msg

# --- Load Data with Loading Indicator ---
def load_data_with_spinner():
    """Load data with a spinner"""
    with st.spinner('🔄 Connecting to database and loading data...'):
        return get_crude_oil_data()

# --- Database Connection Test Function ---
def test_database_connection():
    """Test database connection and return detailed diagnostics"""
    server = r'CLAIRE-NAMUSOKE\SQLEXPRESS'
    database = 'CrudeOilTrade'
    
    results = []
    
    # Test 1: SQLAlchemy Connection
    try:
        params = urllib.parse.quote_plus(
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={server};"
            f"DATABASE={database};"
            f"Trusted_Connection=yes;"
            f"Encrypt=yes;"
            f"TrustServerCertificate=yes;"
        )
        connection_string = f"mssql+pyodbc:///?odbc_connect={params}"
        
        engine = create_engine(connection_string, connect_args={"timeout": 10})
        test_query = "SELECT 1 as test"
        pd.read_sql(test_query, engine)
        engine.dispose()
        results.append("✅ SQLAlchemy connection: SUCCESS")
    except Exception as e:
        results.append(f"❌ SQLAlchemy connection: FAILED - {str(e)}")
    
    # Test 2: Direct pyodbc Connection
    try:
        pyodbc_conn_string = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={server};"
            f"DATABASE={database};"
            f"Trusted_Connection=yes;"
            f"Encrypt=yes;"
            f"TrustServerCertificate=yes;"
            f"Connection Timeout=10;"
        )
        
        conn = pyodbc.connect(pyodbc_conn_string)
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        conn.close()
        results.append("✅ Direct pyodbc connection: SUCCESS")
    except Exception as e:
        results.append(f"❌ Direct pyodbc connection: FAILED - {str(e)}")
    
    return results

# --- OpenAI Connection Test Function ---
def test_openai_connection():
    """Test OpenAI API connection and return detailed diagnostics"""
    results = []
    
    # Test 1: Check if API key is set
    api_key = None
    key_source = ""
    
    # Try Streamlit secrets first
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        key_source = "Streamlit secrets"
    except:
        # Fall back to environment variable
        api_key = os.getenv('OPENAI_API_KEY')
        key_source = "environment variable"
    
    if not api_key:
        results.append("❌ OpenAI API Key: NOT SET in secrets or environment variables")
        results.append("💡 Add to Streamlit secrets or set: $env:OPENAI_API_KEY = 'your-key-here'")
        return results
    else:
        # Mask the key for security (show only first 7 and last 4 characters)
        masked_key = f"{api_key[:7]}...{api_key[-4:]}"
        results.append(f"✅ OpenAI API Key: FOUND from {key_source} ({masked_key})")
    
    # Test 2: Check if OpenAI library is available
    if not OPENAI_AVAILABLE:
        results.append("❌ OpenAI Library: NOT INSTALLED")
        results.append("💡 Install with: pip install openai")
        return results
    else:
        results.append("✅ OpenAI Library: AVAILABLE")
    
    # Test 3: Test API connection with a simple request
    try:
        test_client = OpenAI(api_key=api_key)
        
        # Test with a very simple completion
        response = test_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello"}
            ],
            max_tokens=10,
            timeout=10
        )
        
        if response and response.choices:
            results.append("✅ OpenAI API Connection: SUCCESS")
            results.append(f"🤖 Test Response: {response.choices[0].message.content.strip()}")
            results.append(f"📊 Model Used: {response.model}")
        else:
            results.append("❌ OpenAI API Connection: FAILED - Empty response")
            
    except Exception as e:
        error_msg = str(e)
        if "API key" in error_msg.lower():
            results.append("❌ OpenAI API Connection: INVALID API KEY")
        elif "quota" in error_msg.lower() or "billing" in error_msg.lower():
            results.append("❌ OpenAI API Connection: QUOTA/BILLING ISSUE")
            results.append("💡 Check your billing at: https://platform.openai.com/account/billing")
        elif "network" in error_msg.lower() or "timeout" in error_msg.lower():
            results.append("❌ OpenAI API Connection: NETWORK TIMEOUT")
        else:
            results.append(f"❌ OpenAI API Connection: FAILED - {error_msg}")
    
    return results

# --- Load Data ---
# Clear cache and reload data button
if st.button("🔄 Clear Cache & Reload Data", key="clear_cache"):
    st.cache_data.clear()
    st.rerun()

df, error = get_crude_oil_data()

# --- Function to create dark theme plots ---
def create_dark_theme_plot(fig):
    """Apply dark theme to plotly figures"""
    fig.update_layout(
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
        font_color='#ffffff',
        title_font_color='#ffffff',
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            bordercolor='#ffffff',
            font=dict(color='#ffffff')
        ),
        xaxis=dict(
            gridcolor='#333333',
            color='#ffffff'
        ),
        yaxis=dict(
            gridcolor='#333333',
            color='#ffffff'
        )
    )
    return fig

# --- Main Dashboard ---
def main():
    # Header
    st.title("🌍 Global Crude Oil Trade Dashboard (1995-2021)")
    
    # Check for data loading errors
    if error:
        st.error(f"❌ Database connection failed")
        st.code(error, language="text")
        
        st.info("💡 **Troubleshooting Steps:**")
        st.markdown("""
        1. **Check SQL Server Status**: Make sure SQL Server Express is running
        2. **Test Connection**: Use the 'Test Database Connection' button above
        3. **Clear Cache**: Try the 'Clear Cache & Reload Data' button
        4. **Restart Services**: Restart SQL Server Express service if needed
        """)
        
        # Add service status check
        st.subheader("🔧 Quick Diagnostics")
        if st.button("Check SQL Server Services", key="check_services"):
            try:
                import subprocess
                result = subprocess.run(
                    ['powershell', '-Command', 'Get-Service -Name "*SQL*" | Where-Object {$_.Name -like "*SQLEXPRESS*"} | Format-Table -AutoSize'],
                    capture_output=True, text=True, shell=True
                )
                if result.stdout:
                    st.code(result.stdout, language="text")
                else:
                    st.warning("Could not retrieve service status")
            except Exception as e:
                st.error(f"Error checking services: {e}")
        
        return
    
    if df.empty:
        st.warning("⚠️ No data available from the database.")
        return
    
    # Sidebar Controls
    st.sidebar.header(" Dashboard Controls")
    
    # Year selection
    years = sorted(df['Year'].unique())
    year_options = ['All'] + years
    selected_years = st.sidebar.multiselect(
        "📅 Select Years:",
        options=year_options,
        default=['All'] if years else []
    )
    
    # Continent selection
    continents = ['All'] + sorted(df['Continent'].unique())
    selected_continent = st.sidebar.selectbox(
        "🌍 Select Continent:",
        options=continents
    )
    
    # Color scheme selection for accessibility
    color_scheme = st.sidebar.selectbox(
        "🎨 Map Color Scheme:",
        options=['Classic Viridis', 'Colorblind-Friendly', 'High Contrast'],
        help="Choose a color scheme that works best for you"
    )
    
    # Filter data
    filtered_df = df.copy()
    
    if selected_years and 'All' not in selected_years:
        filtered_df = filtered_df[filtered_df['Year'].isin(selected_years)]
        
    if selected_continent != 'All':
        filtered_df = filtered_df[filtered_df['Continent'] == selected_continent]
    
    # Main Content Area
    if filtered_df.empty:
        st.warning("⚠️ No data matches your selection criteria.")
        return
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_records = len(filtered_df)
        st.markdown(f"""
        <div style='text-align: center; padding: 6px;'>
            <p style='color: #ffffff; margin-bottom: 3px; font-size: 0.8rem; font-weight: normal;'>Total Records</p>
            <h2 style='color: #00FFFF; margin: 0; font-size: 1.5rem; text-shadow: 0 0 8px rgba(0,255,255,0.4);'>{total_records:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_trade = filtered_df['TradeValue'].sum()
        
        # Format trade value in billions or trillions
        if total_trade >= 1e12:
            trade_value_formatted = f"${total_trade / 1e12:.2f}T"
        else:
            trade_value_formatted = f"${total_trade / 1e9:.1f}B"
            
        st.markdown(f"""
        <div style='text-align: center; padding: 6px;'>
            <p style='color: #ffffff; margin-bottom: 3px; font-size: 0.8rem; font-weight: normal;'>Total Trade Value</p>
            <h2 style='color: #00FFFF; margin: 0; font-size: 1.5rem; text-shadow: 0 0 8px rgba(0,255,255,0.4);'>{trade_value_formatted}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        total_countries = len(filtered_df['Country'].unique())
        st.markdown(f"""
        <div style='text-align: center; padding: 6px;'>
            <p style='color: #ffffff; margin-bottom: 3px; font-size: 0.8rem; font-weight: normal;'>Countries</p>
            <h2 style='color: #00FFFF; margin: 0; font-size: 1.5rem; text-shadow: 0 0 8px rgba(0,255,255,0.4);'>{total_countries:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        years_count = len(filtered_df['Year'].unique()) if not filtered_df.empty else len(df['Year'].unique())
        st.markdown(f"""
        <div style='text-align: center; padding: 6px;'>
            <p style='color: #ffffff; margin-bottom: 3px; font-size: 0.8rem; font-weight: normal;'>Years Covered</p>
            <h2 style='color: #00FFFF; margin: 0; font-size: 1.5rem; text-shadow: 0 0 8px rgba(0,255,255,0.4);'>{years_count:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Trade Value by Action (Donut Chart)
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader(" Trade Value by Action")
        
        if not filtered_df.empty:
            # Calculate total trade value by action
            action_totals = filtered_df.groupby('Action')['TradeValue'].sum().reset_index()
            
            # Create donut chart with consistent dashboard colors
            fig_donut = px.pie(
                action_totals,
                values='TradeValue',
                names='Action',
                title="Import vs Export Distribution",
                hole=0.4  # This creates the donut effect
            )
            
            # Apply dark theme first
            fig_donut = create_dark_theme_plot(fig_donut)
            
            # Manually set colors to match dashboard theme (after dark theme to ensure they stick)
            colors = []
            for action in action_totals['Action']:
                if action == 'Import':
                    colors.append('#FFA500')  # Orange for imports
                elif action == 'Export':
                    colors.append('#00FFFF')  # Cyan for exports
                else:
                    colors.append('#FFFFFF')  # White fallback
            
            # Update traces for better visibility and apply manual colors
            fig_donut.update_traces(
                textposition='inside',
                textinfo='percent',  # Only show percentages, remove labels
                textfont_color='white',
                textfont_size=16,  # Slightly larger since no labels
                marker=dict(
                    colors=colors,
                    line=dict(color='#FFFFFF', width=2)
                )
            )
            
            # Additional styling for donut chart
            fig_donut.update_layout(
                height=280,
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.01,
                    bgcolor='rgba(0,0,0,0)',
                    bordercolor='rgba(0,0,0,0)',  # No border
                    borderwidth=0,  # No border width
                    font=dict(color='white', size=10)
                ),
                title=dict(
                    font=dict(size=12, color='white'),
                    x=0.5,
                    xanchor='center'
                ),
                margin=dict(l=5, r=5, t=20, b=5)
            )
            
            st.plotly_chart(fig_donut, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("No data available for action analysis.")
    
    with col2:
        # Action insights and statistics
        st.subheader("📊 Action Analysis Insights")
        
        if not filtered_df.empty:
            # Calculate statistics
            import_total = filtered_df[filtered_df['Action'] == 'Import']['TradeValue'].sum()
            export_total = filtered_df[filtered_df['Action'] == 'Export']['TradeValue'].sum()
            total_trade = import_total + export_total
            
            # Format values
            if import_total >= 1e12:
                import_formatted = f"${import_total / 1e12:.2f}T"
            else:
                import_formatted = f"${import_total / 1e9:.1f}B"
                
            if export_total >= 1e12:
                export_formatted = f"${export_total / 1e12:.2f}T"
            else:
                export_formatted = f"${export_total / 1e9:.1f}B"
            
            # Display metrics in a clean layout
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown(f"""
                <div style='text-align: center; padding: 4px; border: 1px solid #FFA500; border-radius: 5px; margin: 1px;'>
                    <h6 style='color: #FFA500; margin: 0; font-size: 0.7rem;'>📥 Imports</h6>
                    <h4 class='action-value' style='margin: 1px 0; font-size: 1.1rem;'>{import_formatted}</h4>
                    <p style='color: #ffffff; margin: 0; font-size: 0.6rem;'>{(import_total/total_trade*100):.1f}% of total trade</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_b:
                st.markdown(f"""
                <div style='text-align: center; padding: 4px; border: 1px solid #00FFFF; border-radius: 5px; margin: 1px;'>
                    <h6 style='color: #00FFFF; margin: 0; font-size: 0.7rem;'>📤 Exports</h6>
                    <h4 class='action-value' style='margin: 1px 0; font-size: 1.1rem;'>{export_formatted}</h4>
                    <p style='color: #ffffff; margin: 0; font-size: 0.6rem;'>{(export_total/total_trade*100):.1f}% of total trade</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Trade balance analysis and Key insights in same row
            st.markdown("### 🏦 Trade Balance & Key Insights")
            
            trade_balance = export_total - import_total
            
            if trade_balance > 0:
                balance_color = "#00FF00"  # Green for positive
                balance_text = "Trade Surplus"
                balance_icon = ""
            else:
                balance_color = "#FF4444"  # Red for negative
                balance_text = "Trade Deficit"
                balance_icon = ""
            
            if abs(trade_balance) >= 1e12:
                balance_formatted = f"${abs(trade_balance) / 1e12:.2f}T"
            else:
                balance_formatted = f"${abs(trade_balance) / 1e9:.1f}B"
            
            # Create two columns for balance and insights
            col_balance, col_insights = st.columns(2)
            
            with col_balance:
                st.markdown(f"""
                <div style='text-align: center; padding: 4px; border: 1px solid {balance_color}; border-radius: 5px; margin: 1px;'>
                    <h6 style='color: {balance_color}; margin: 0; font-size: 0.7rem;'>{balance_icon} {balance_text}</h6>
                    <h4 class='action-value' style='margin: 1px 0; font-size: 1.1rem;'>{balance_formatted}</h4>
                    <p style='color: #ffffff; margin: 0; font-size: 0.6rem;'>
                        {'Exports > Imports' if trade_balance > 0 else 'Imports > Exports'}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_insights:
                st.markdown("**💡 Key Insights:**")
                
                # Calculate number of records for each action
                import_records = len(filtered_df[filtered_df['Action'] == 'Import'])
                export_records = len(filtered_df[filtered_df['Action'] == 'Export'])
                
                # Calculate average transaction value
                avg_import_value = import_total / import_records if import_records > 0 else 0
                avg_export_value = export_total / export_records if export_records > 0 else 0
                
                insights = []
                
                if import_records > export_records:
                    insights.append(" More import transactions")
                elif export_records > import_records:
                    insights.append(" More export transactions")
                else:
                    insights.append(" Equal transactions")
                
                if avg_export_value > avg_import_value:
                    insights.append(" Higher avg export value")
                elif avg_import_value > avg_export_value:
                    insights.append(" Higher avg import value")
                
                if trade_balance > total_trade * 0.1:
                    insights.append(" Strong export performance")
                elif trade_balance < -total_trade * 0.1:
                    insights.append(" High import demand")
                else:
                    insights.append(" Balanced trade activity")
                
                # Display insights in compact format
                for insight in insights:
                    st.markdown(f"<p style='font-size: 0.75rem; margin: 2px 0; color: #ffffff;'>• {insight}</p>", unsafe_allow_html=True)
        else:
            st.info("No data available for detailed analysis.")
    
    # Top Import/Export Tables Row
    st.subheader(" Top 10 Countries by Import and Export")
    col1, col2 = st.columns(2)
    
    with col1:
        # Get top importers
        importers_df = filtered_df[filtered_df['Action'] == 'Import']
        if not importers_df.empty:
            top_importers = importers_df.groupby('Country')['TradeValue'].sum().reset_index()
            total_imports = top_importers['TradeValue'].sum()
            top_importers['Import (%)'] = (top_importers['TradeValue'] / total_imports * 100).round(2)
            top_importers = top_importers.nlargest(10, 'TradeValue')
            top_importers_display = top_importers[['Country', 'Import (%)']].reset_index(drop=True)
            top_importers_display = top_importers_display.rename(columns={'Country': 'Importers'})
            top_importers_display.index += 1
            
            # Apply orange gradient styling using custom function
            def color_importers(val):
                max_val = top_importers_display['Import (%)'].max()
                min_val = top_importers_display['Import (%)'].min()
                
                if isinstance(val, (int, float)) and max_val > min_val:
                    # Normalize the value (0 to 1)
                    normalized = (val - min_val) / (max_val - min_val)
                    # Create orange gradient - light to dark orange
                    if normalized > 0.8:
                        return 'background-color: #FF8C00; color: black; font-weight: normal'  # Dark orange
                    elif normalized > 0.6:
                        return 'background-color: #FFA500; color: black; font-weight: normal'  # Orange
                    elif normalized > 0.4:
                        return 'background-color: #FFB84D; color: black; font-weight: normal'  # Light orange
                    elif normalized > 0.2:
                        return 'background-color: #FFCC80; color: black; font-weight: normal'  # Lighter orange
                    else:
                        return 'background-color: #FFE0B3; color: black; font-weight: normal'  # Very light orange
                return ''
            
            styled_importers = top_importers_display.style.map(
                color_importers, subset=['Import (%)']
            ).format({'Import (%)': '{:.2f}%'}).set_properties(**{
                'border': '1px solid #555555'
            }).set_table_styles([
                {'selector': 'th', 'props': [('background-color', '#333333'), ('color', 'white'), ('border', '1px solid #555555')]}
            ])
            
            st.dataframe(styled_importers, width='stretch')
        else:
            st.info("No import data available for the selected filters.")
    
    with col2:
        # Get top exporters
        exporters_df = filtered_df[filtered_df['Action'] == 'Export']
        if not exporters_df.empty:
            top_exporters = exporters_df.groupby('Country')['TradeValue'].sum().reset_index()
            total_exports = top_exporters['TradeValue'].sum()
            top_exporters['Export (%)'] = (top_exporters['TradeValue'] / total_exports * 100).round(2)
            top_exporters = top_exporters.nlargest(10, 'TradeValue')
            top_exporters_display = top_exporters[['Country', 'Export (%)']].reset_index(drop=True)
            top_exporters_display = top_exporters_display.rename(columns={'Country': 'Exporters'})
            top_exporters_display.index += 1
            
            # Apply cyan gradient styling using custom function
            def color_exporters(val):
                max_val = top_exporters_display['Export (%)'].max()
                min_val = top_exporters_display['Export (%)'].min()
                
                if isinstance(val, (int, float)) and max_val > min_val:
                    # Normalize the value (0 to 1)
                    normalized = (val - min_val) / (max_val - min_val)
                    # Create cyan gradient - light to dark cyan
                    if normalized > 0.8:
                        return 'background-color: #00CED1; color: black; font-weight: normal'  # Dark turquoise
                    elif normalized > 0.6:
                        return 'background-color: #00FFFF; color: black; font-weight: normal'  # Cyan
                    elif normalized > 0.4:
                        return 'background-color: #40E0D0; color: black; font-weight: normal'  # Turquoise
                    elif normalized > 0.2:
                        return 'background-color: #7FFFD4; color: black; font-weight: normal'  # Aquamarine
                    else:
                        return 'background-color: #B0E0E6; color: black; font-weight: normal'  # Powder blue
                return ''
            
            styled_exporters = top_exporters_display.style.map(
                color_exporters, subset=['Export (%)']
            ).format({'Export (%)': '{:.2f}%'}).set_properties(**{
                'border': '1px solid #555555'
            }).set_table_styles([
                {'selector': 'th', 'props': [('background-color', '#333333'), ('color', 'white'), ('border', '1px solid #555555')]}
            ])
            
            st.dataframe(styled_exporters, width='stretch')
        else:
            st.info("No export data available for the selected filters.")
    
    # Trade Analysis Charts - Side by Side
    col1, col2 = st.columns(2)
    
    with col1:
        if not filtered_df.empty:
            # Use existing continent data
            df_continent = filtered_df.copy()
            
            # Calculate continent totals for imports and exports
            continent_imports = df_continent[df_continent['Action'] == 'Import'].groupby('Continent')['TradeValue'].sum().reset_index()
            continent_imports.columns = ['Continent', 'Imports']
            
            continent_exports = df_continent[df_continent['Action'] == 'Export'].groupby('Continent')['TradeValue'].sum().reset_index()
            continent_exports.columns = ['Continent', 'Exports']
            
            # Merge imports and exports
            continent_data = pd.merge(continent_imports, continent_exports, on='Continent', how='outer').fillna(0)
            
            # Sort by total trade value
            continent_data['Total'] = continent_data['Imports'] + continent_data['Exports']
            continent_data = continent_data.sort_values('Total', ascending=True)
            
            if not continent_data.empty:
                # Create vertical bar chart using plotly
                fig_continent = go.Figure()
                
                # Add imports bar (orange)
                fig_continent.add_trace(go.Bar(
                    name='Imports',
                    x=continent_data['Continent'],
                    y=continent_data['Imports'],
                    marker_color='#FFA500'
                ))
                
                # Add exports bar (cyan)
                fig_continent.add_trace(go.Bar(
                    name='Exports',
                    x=continent_data['Continent'],
                    y=continent_data['Exports'],
                    marker_color='#00FFFF'
                ))
                
                # Update layout with dark theme
                fig_continent.update_layout(
                    title={
                        'text': 'Trade Volume by Continent',
                        'x': 0.5,
                        'xanchor': 'center',
                        'font': {'color': 'white', 'size': 12}
                    },
                    xaxis=dict(
                        title='Continent',
                        gridcolor='#333333',
                        color='white',
                        title_font=dict(size=10)
                    ),
                    yaxis=dict(
                        title='Trade Value',
                        gridcolor='#333333',
                        color='white',
                        tickformat='$.1s',
                        title_font=dict(size=10)
                    ),
                    plot_bgcolor='#000000',
                    paper_bgcolor='#000000',
                    font=dict(color='white', size=10),
                    legend=dict(
                        bgcolor='rgba(0,0,0,0)',
                        bordercolor='rgba(0,0,0,0)',  # No border
                        borderwidth=0,  # No border width
                        font=dict(size=10)
                    ),
                    barmode='group',
                    height=280,
                    margin=dict(l=40, r=20, t=40, b=40)
                )
                
                st.plotly_chart(fig_continent, use_container_width=True, config={'displayModeBar': False})
            else:
                st.info("No continent data available for the selected filters.")
        else:
            st.info("No data available for continent overview.")
    
    with col2:
        
        
        if not filtered_df.empty:
            # Group by year and action to get time series data
            time_series_data = filtered_df.groupby(['Year', 'Action'])['TradeValue'].sum().reset_index()
            
            # Pivot to get imports and exports as separate columns
            time_series_pivot = time_series_data.pivot(index='Year', columns='Action', values='TradeValue').reset_index()
            time_series_pivot = time_series_pivot.fillna(0)
            
            # Convert to trillions
            if 'Import' in time_series_pivot.columns:
                time_series_pivot['Import_Trillions'] = time_series_pivot['Import'] / 1e12
            else:
                time_series_pivot['Import_Trillions'] = 0
                
            if 'Export' in time_series_pivot.columns:
                time_series_pivot['Export_Trillions'] = time_series_pivot['Export'] / 1e12
            else:
                time_series_pivot['Export_Trillions'] = 0
            
            # Create line chart
            fig_time_series = go.Figure()
            
            # Add imports line (orange)
            fig_time_series.add_trace(go.Scatter(
                x=time_series_pivot['Year'],
                y=time_series_pivot['Import_Trillions'],
                mode='lines+markers',
                name='Imports',
                line=dict(color='#FFA500', width=3),
                marker=dict(color='#FFA500', size=6)
            ))
            
            # Add exports line (cyan)
            fig_time_series.add_trace(go.Scatter(
                x=time_series_pivot['Year'],
                y=time_series_pivot['Export_Trillions'],
                mode='lines+markers',
                name='Exports',
                line=dict(color='#00FFFF', width=3),
                marker=dict(color='#00FFFF', size=6)
            ))
            
            # Update layout with dark theme
            fig_time_series.update_layout(
                title={
                    'text': 'Trade Trends Over Time',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'color': 'white', 'size': 12}
                },
                xaxis=dict(
                    title='Year',
                    gridcolor='#333333',
                    color='white',
                    showgrid=True,
                    title_font=dict(size=10)
                ),
                yaxis=dict(
                    title='Trade Value (Trillion USD)',
                    gridcolor='#333333',
                    color='white',
                    showgrid=True,
                    tickformat='$.2f',
                    title_font=dict(size=10)
                ),
                plot_bgcolor='#000000',
                paper_bgcolor='#000000',
                font=dict(color='white', size=10),
                legend=dict(
                    bgcolor='rgba(0,0,0,0)',
                    bordercolor='rgba(0,0,0,0)',  # No border
                    borderwidth=0,  # No border width
                    x=0.02,
                    y=0.98,
                    font=dict(size=10)
                ),
                height=280,
                margin=dict(l=40, r=20, t=40, b=40),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_time_series, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("No data available for time series chart.")
    
    # Charts Row 1: World Map and Top Countries
    col1, col2 = st.columns([2, 1])
    
    with col1:
        
        
        # World map using plotly with dark theme
        country_totals = filtered_df.groupby('Country')['TradeValue'].sum().reset_index()
        
        # Create comprehensive country name to ISO-3 mapping
        country_iso_mapping = {
            # Major Global Powers
            'United States': 'USA', 'China': 'CHN', 'Germany': 'DEU', 'Japan': 'JPN',
            'United Kingdom': 'GBR', 'France': 'FRA', 'Italy': 'ITA', 'Brazil': 'BRA',
            'Canada': 'CAN', 'Russia': 'RUS', 'India': 'IND', 'Australia': 'AUS',
            'Spain': 'ESP', 'Mexico': 'MEX', 'Indonesia': 'IDN', 'Netherlands': 'NLD',
            'Saudi Arabia': 'SAU', 'Turkey': 'TUR', 'Taiwan': 'TWN', 'Belgium': 'BEL',
            'Poland': 'POL', 'Argentina': 'ARG', 'Ireland': 'IRL', 'Austria': 'AUT',
            'Israel': 'ISR', 'Thailand': 'THA', 'Philippines': 'PHL', 'Chile': 'CHL',
            'Finland': 'FIN', 'Bangladesh': 'BGD', 'Vietnam': 'VNM', 'Malaysia': 'MYS',
            'Singapore': 'SGP', 'Denmark': 'DNK', 'Norway': 'NOR', 'New Zealand': 'NZL',
            'Hong Kong': 'HKG', 'Sweden': 'SWE', 'Czech Republic': 'CZE', 'United Arab Emirates': 'ARE',
            'Portugal': 'PRT', 'Romania': 'ROU', 'Peru': 'PER', 'Greece': 'GRC',
            'Iraq': 'IRQ', 'Kazakhstan': 'KAZ', 'Hungary': 'HUN', 'Qatar': 'QAT',
            'Kuwait': 'KWT', 'Slovakia': 'SVK', 'Ecuador': 'ECU', 'South Korea': 'KOR',
            'Venezuela': 'VEN', 'Colombia': 'COL', 'Iran': 'IRN', 'Bulgaria': 'BGR',
            'Croatia': 'HRV', 'Lithuania': 'LTU', 'Slovenia': 'SVN', 'Luxembourg': 'LUX',
            'Latvia': 'LVA', 'Estonia': 'EST', 'Cyprus': 'CYP', 'Malta': 'MLT',
            
            # Comprehensive African Countries
            'Nigeria': 'NGA', 'Egypt': 'EGY', 'South Africa': 'ZAF', 'Algeria': 'DZA',
            'Morocco': 'MAR', 'Angola': 'AGO', 'Libya': 'LBY', 'Ghana': 'GHA',
            'Kenya': 'KEN', 'Ethiopia': 'ETH', 'Tanzania': 'TZA', 'Uganda': 'UGA',
            'Cameroon': 'CMR', 'Madagascar': 'MDG', 'Mali': 'MLI', 'Burkina Faso': 'BFA',
            'Niger': 'NER', 'Malawi': 'MWI', 'Zambia': 'ZMB', 'Senegal': 'SEN',
            'Somalia': 'SOM', 'Chad': 'TCD', 'Guinea': 'GIN', 'Rwanda': 'RWA',
            'Benin': 'BEN', 'Tunisia': 'TUN', 'Burundi': 'BDI', 'South Sudan': 'SSD',
            'Togo': 'TGO', 'Sierra Leone': 'SLE', 'Libya': 'LBY', 'Liberia': 'LBR',
            'Central African Republic': 'CAF', 'Mauritania': 'MRT', 'Eritrea': 'ERI',
            'Gambia': 'GMB', 'Botswana': 'BWA', 'Namibia': 'NAM', 'Gabon': 'GAB',
            'Lesotho': 'LSO', 'Guinea-Bissau': 'GNB', 'Equatorial Guinea': 'GNQ',
            'Mauritius': 'MUS', 'Eswatini': 'SWZ', 'Djibouti': 'DJI', 'Comoros': 'COM',
            'Cape Verde': 'CPV', 'Sao Tome and Principe': 'STP', 'Seychelles': 'SYC',
            'Zimbabwe': 'ZWE', 'Mozambique': 'MOZ', 'Ivory Coast': 'CIV', 'Cote d\'Ivoire': 'CIV',
            
            # Additional countries that might appear in trade data
            'Congo': 'COG', 'Democratic Republic of the Congo': 'COD', 'Sudan': 'SDN'
        }
        
        # Add ISO codes to country data
        country_totals['iso_alpha'] = country_totals['Country'].map(country_iso_mapping)
        
        # Debug: Show countries without ISO mapping
        missing_iso = country_totals[country_totals['iso_alpha'].isna()]
        if not missing_iso.empty and selected_continent == 'Africa':
            with st.expander("🔍 Debug: Countries Missing ISO Codes"):
                st.write(f"Found {len(missing_iso)} countries without ISO mapping:")
                st.dataframe(missing_iso[['Country', 'TradeValue']].sort_values('TradeValue', ascending=False))
        
        # For continent-specific views, add all continent countries with base values to fill gaps
        if selected_continent != 'All':
            # Define complete country lists for each continent
            all_continent_countries = {
                'Africa': [
                    'Nigeria', 'Egypt', 'South Africa', 'Algeria', 'Morocco', 'Angola', 'Libya', 
                    'Ghana', 'Kenya', 'Ethiopia', 'Tanzania', 'Uganda', 'Cameroon', 'Madagascar',
                    'Mali', 'Burkina Faso', 'Niger', 'Malawi', 'Zambia', 'Senegal', 'Somalia',
                    'Chad', 'Guinea', 'Rwanda', 'Benin', 'Tunisia', 'Burundi', 'South Sudan',
                    'Togo', 'Sierra Leone', 'Liberia', 'Central African Republic', 'Mauritania',
                    'Eritrea', 'Gambia', 'Botswana', 'Namibia', 'Gabon', 'Lesotho', 'Guinea-Bissau',
                    'Equatorial Guinea', 'Mauritius', 'Eswatini', 'Djibouti', 'Comoros', 'Cape Verde',
                    'Sao Tome and Principe', 'Seychelles', 'Zimbabwe', 'Mozambique', 'Ivory Coast',
                    'Congo', 'Democratic Republic of the Congo', 'Sudan'
                ],
                'Europe': [
                    'Germany', 'United Kingdom', 'France', 'Italy', 'Spain', 'Netherlands', 'Belgium',
                    'Poland', 'Austria', 'Ireland', 'Finland', 'Denmark', 'Norway', 'Sweden',
                    'Czech Republic', 'Portugal', 'Romania', 'Greece', 'Hungary', 'Slovakia',
                    'Bulgaria', 'Croatia', 'Lithuania', 'Slovenia', 'Luxembourg', 'Latvia',
                    'Estonia', 'Cyprus', 'Malta'
                ],
                'Asia': [
                    'China', 'Japan', 'India', 'Indonesia', 'Thailand', 'Philippines', 'Malaysia',
                    'Singapore', 'Vietnam', 'Bangladesh', 'South Korea', 'Taiwan', 'Hong Kong',
                    'Israel', 'United Arab Emirates', 'Saudi Arabia', 'Qatar', 'Kuwait', 'Iran',
                    'Iraq', 'Kazakhstan', 'Turkey'
                ],
                'North America': [
                    'United States', 'Canada', 'Mexico'
                ],
                'South America': [
                    'Brazil', 'Argentina', 'Chile', 'Peru', 'Colombia', 'Venezuela', 'Ecuador'
                ],
                'Oceania': [
                    'Australia', 'New Zealand', 'Papua New Guinea', 'Fiji', 'Solomon Islands',
                    'New Caledonia', 'French Polynesia', 'Vanuatu', 'Samoa', 'Kiribati',
                    'Federated States of Micronesia', 'Tonga', 'Marshall Islands', 'Palau',
                    'Cook Islands', 'Nauru', 'Tuvalu', 'Niue', 'American Samoa', 'Guam',
                    'Northern Mariana Islands', 'Tokelau', 'Wallis and Futuna'
                ]
            }
            
            if selected_continent in all_continent_countries:
                # Get all countries that should be in this continent
                continent_countries = all_continent_countries[selected_continent]
                
                # Create DataFrame with all continent countries
                all_continent_df = pd.DataFrame({
                    'Country': continent_countries,
                    'TradeValue': 0  # Default value for countries with no trade data
                })
                
                # Add ISO codes
                all_continent_df['iso_alpha'] = all_continent_df['Country'].map(country_iso_mapping)
                all_continent_df = all_continent_df.dropna(subset=['iso_alpha'])
                
                # Merge with actual trade data (actual data will override the 0 values)
                country_totals_with_iso = country_totals.dropna(subset=['iso_alpha'])
                
                # Combine: start with all countries, then update with actual trade data
                final_countries = all_continent_df.copy()
                
                # Update with actual trade data where available
                for idx, row in country_totals_with_iso.iterrows():
                    mask = final_countries['iso_alpha'] == row['iso_alpha']
                    if mask.any():
                        final_countries.loc[mask, 'TradeValue'] = row['TradeValue']
                        final_countries.loc[mask, 'Country'] = row['Country']
                    else:
                        # Add countries that are in trade data but not in our complete list
                        new_row = pd.DataFrame({
                            'Country': [row['Country']],
                            'TradeValue': [row['TradeValue']],
                            'iso_alpha': [row['iso_alpha']]
                        })
                        final_countries = pd.concat([final_countries, new_row], ignore_index=True)
                
                mapped_countries = final_countries
            else:
                # For continents without complete lists, use original method
                mapped_countries = country_totals.dropna(subset=['iso_alpha'])
        else:
            # For 'All' view, use original method
            mapped_countries = country_totals.dropna(subset=['iso_alpha'])
        
        # Dynamic title based on continent selection
        if selected_continent == 'All':
            map_title = "All Trades: Global View (Billion USD)"
        else:
            map_title = f"All Trades: {selected_continent} View (Billion USD)"
        
        # Adjust color scale to better show continent-specific data
        if len(mapped_countries) > 0:
            # Separate countries with trade data from those without
            has_trade = mapped_countries[mapped_countries['TradeValue'] > 0]
            
            if len(has_trade) > 0:
                # Use range starting from smallest positive value for better visualization
                min_val = has_trade['TradeValue'].min()
                max_val = mapped_countries['TradeValue'].max()
                
                # Define different color schemes for accessibility
                if color_scheme == 'Classic Viridis':
                    color_scale = [
                        [0, '#404040'],        # Medium gray for zero values (distinguishable from Viridis)
                        [0.001, '#440154'],   # Deep purple (Viridis start)
                        [0.25, '#31688e'],    # Blue
                        [0.5, '#35b779'],     # Green 
                        [0.75, '#90d743'],    # Light green
                        [1, '#fde725']        # Bright yellow (Viridis end)
                    ]
                    legend_text = "Gray (no data) → **Purple** (low) → **Blue** → **Green** → **Yellow** (high) - Classic Viridis colors"
                elif color_scheme == 'Colorblind-Friendly':
                    color_scale = [
                        [0, '#404040'],        # Medium gray for zero values
                        [0.001, '#1f77b4'],   # Blue for low values
                        [0.3, '#17becf'],      # Light blue for medium-low values
                        [0.6, '#2ca02c'],      # Green for medium values
                        [0.8, '#ff7f0e'],      # Orange for high values
                        [1, '#d62728']         # Red for maximum values
                    ]
                    legend_text = "Gray (no data) → **Blue** (low) → **Light Blue** → **Green** (medium) → **Orange** → **Red** (high)"
                else:  # High Contrast
                    color_scale = [
                        [0, '#000000'],        # Black for zero values
                        [0.001, '#FFFFFF'],   # White for any positive values
                        [0.5, '#FFFF00'],      # Yellow for medium values
                        [1, '#FF0000']         # Bright red for maximum values
                    ]
                    legend_text = "**Black** (no data) → **White** (low) → **Yellow** (medium) → **Red** (high)"
                
                fig_map = px.choropleth(
                    mapped_countries,
                    locations='iso_alpha',
                    color='TradeValue',
                    locationmode='ISO-3',
                    color_continuous_scale=color_scale,
                    title=map_title,
                    labels={'TradeValue': 'Trade Value ($)'},
                    hover_data={'Country': True},
                    range_color=[0, max_val]  # Include zero in range
                )
            else:
                # All countries have zero trade
                fig_map = px.choropleth(
                    mapped_countries,
                    locations='iso_alpha',
                    color='TradeValue',
                    locationmode='ISO-3',
                    color_continuous_scale='Greys',
                    title=map_title,
                    labels={'TradeValue': 'Trade Value ($)'},
                    hover_data={'Country': True}
                )
        else:
            # Fallback if no countries have ISO codes
            fig_map = px.choropleth(
                title=f"{map_title} - No data available with valid country codes"
            )
        
        # Continental zoom and focus
        if selected_continent != 'All':
            # Define continent scopes for geographical focusing
            continent_scopes = {
                'Europe': {'scope': 'europe'},
                'Asia': {'scope': 'asia'},
                'North America': {'scope': 'north america'},
                'South America': {'scope': 'south america'},
                'Africa': {'scope': 'africa'},
                'Oceania': {
                    'scope': 'world',
                    'projection_type': 'natural earth',
                    'center': {'lat': -25, 'lon': 140},
                    'projection_scale': 4
                }
            }
            
            if selected_continent in continent_scopes:
                geo_config = continent_scopes[selected_continent]
                fig_map.update_geos(**geo_config)
        
        fig_map.update_layout(
            height=380,
            title_font=dict(size=12),
            margin=dict(l=0, r=0, t=30, b=0)
        )
        fig_map = create_dark_theme_plot(fig_map)
        fig_map.update_geos(
            bgcolor='#000000',
            showframe=False,
            showcoastlines=True,
            coastlinecolor='#333333'
        )
        st.plotly_chart(fig_map, use_container_width=True, config={'displayModeBar': False})
        
        # Add explanation for continental view
        if selected_continent != 'All':
            if 'legend_text' in locals():
                st.caption(f"💡 **Map Legend**: {legend_text}")
            else:
                st.caption("💡 **Map Legend**: Countries with no trade data appear in the background. Colored countries show relative trade values.")
    
    with col2:
        # Dynamic chart: Continent view by default, Top 10 countries when filtered by continent
        if selected_continent == 'All':
            # Group by continent for continent view
            continent_totals = filtered_df.groupby('Continent')['TradeValue'].sum().reset_index()
            continent_totals = continent_totals.sort_values('TradeValue', ascending=True)
            
            fig_dynamic = px.bar(
                continent_totals,
                x='TradeValue',
                y='Continent',
                orientation='h',
                title="Trade Value by Continent",
                labels={'TradeValue': 'Trade Value', 'Continent': 'Continent'},
                color='TradeValue',
                color_continuous_scale='Viridis'
            )
            fig_dynamic.update_layout(
                yaxis={'categoryorder': 'total ascending'}, 
                height=380,
                showlegend=False,
                title_font=dict(size=12),
                margin=dict(l=60, r=20, t=40, b=40),
                font=dict(size=10)
            )
            fig_dynamic.update_coloraxes(showscale=False)
            
        else:
            st.subheader(f"🔝 Top 10 Countries in {selected_continent}")
            
            # Group by country for country view (when continent is filtered)
            country_totals = filtered_df.groupby('Country')['TradeValue'].sum().reset_index()
            top_10_countries = country_totals.nlargest(10, 'TradeValue')
            
            fig_dynamic = px.bar(
                top_10_countries,
                x='TradeValue',
                y='Country',
                orientation='h',
                title=f"Top 10 Countries by Trade Value",
                labels={'TradeValue': 'Trade Value', 'Country': 'Country'},
                color='TradeValue',
                color_continuous_scale='Viridis'
            )
            fig_dynamic.update_layout(
                yaxis={'categoryorder': 'total ascending'}, 
                height=380,
                showlegend=False,
                title_font=dict(size=12),
                margin=dict(l=60, r=20, t=40, b=40),
                font=dict(size=10)
            )
            fig_dynamic.update_coloraxes(showscale=False)
        
        # Apply dark theme and display
        fig_dynamic = create_dark_theme_plot(fig_dynamic)
        st.plotly_chart(fig_dynamic, use_container_width=True, config={'displayModeBar': False})
    
    # --- YoY Percentage Change Analysis ---
    
    # Calculate year-over-year percentage change
    if len(filtered_df) > 0:
        # Group by year and calculate total trade value per year
        yearly_totals = filtered_df.groupby('Year')['TradeValue'].sum().reset_index()
        yearly_totals = yearly_totals.sort_values('Year')
        
        # Calculate YoY percentage change
        yearly_totals['YoY_Change_Pct'] = yearly_totals['TradeValue'].pct_change() * 100
        
        # Remove first year (no previous year to compare)
        yoy_data = yearly_totals.dropna()
        
        if len(yoy_data) > 0:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Create YoY change chart
                fig_yoy = go.Figure()
                
                # Add bars with conditional coloring (cyan for positive, red for negative)
                colors = ['#00FFFF' if change >= 0 else '#FF4444' for change in yoy_data['YoY_Change_Pct']]
                
                fig_yoy.add_trace(go.Bar(
                    x=yoy_data['Year'],
                    y=yoy_data['YoY_Change_Pct'],
                    marker_color=colors,
                    name='YoY % Change',
                    text=[f'{change:+.1f}%' for change in yoy_data['YoY_Change_Pct']],
                    textposition='outside'
                ))
                
                # Add zero line for reference
                fig_yoy.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.7)
                
                # Update layout with dark theme
                fig_yoy.update_layout(
                    title={
                        'text': 'Year-over-Year Percentage Change in Trade Value',
                        'x': 0.5,
                        'xanchor': 'center',
                        'font': {'color': 'white', 'size': 12}
                    },
                    xaxis=dict(
                        title='Year',
                        gridcolor='#333333',
                        color='white',
                        showgrid=False,
                        title_font=dict(size=10)
                    ),
                    yaxis=dict(
                        title='YoY Change (%)',
                        gridcolor='#333333',
                        color='white',
                        showgrid=False,
                        tickformat='.1f',
                        title_font=dict(size=10)
                    ),
                    plot_bgcolor='#000000',
                    paper_bgcolor='#000000',
                    font=dict(color='white', size=10),
                    showlegend=False,
                    height=280,
                    margin=dict(l=40, r=20, t=40, b=40)
                )
                
                st.plotly_chart(fig_yoy, use_container_width=True, config={'displayModeBar': False})

            with col2:
                st.markdown("### 📊 Key Insights")

                # Calculate statistics
                avg_growth = yoy_data['YoY_Change_Pct'].mean()
                max_growth = yoy_data['YoY_Change_Pct'].max()
                min_growth = yoy_data['YoY_Change_Pct'].min()
                max_growth_year = yoy_data[yoy_data['YoY_Change_Pct'] == max_growth]['Year'].iloc[0]
                min_growth_year = yoy_data[yoy_data['YoY_Change_Pct'] == min_growth]['Year'].iloc[0]

                # Render custom styled metrics with conditional coloring for positive/negative values
                avg_growth_class = 'yoy-positive' if avg_growth >= 0 else 'yoy-negative'
                avg_growth_sign = '+' if avg_growth >= 0 else ''
                
                st.markdown(f"""
                <div style='display:flex; gap:10px; flex-wrap:wrap;'>
                    <div class='yoy-metric-box'>
                        <div class='yoy-metric-label'>Average YoY Growth</div>
                        <div class='{avg_growth_class}'>{avg_growth_sign}{avg_growth:.1f}%</div>
                    </div>
                    <div class='yoy-metric-box' style='min-width:160px;'>
                        <div class='yoy-metric-label'>Best Year</div>
                        <div class='yoy-metric-large'><span class='yoy-year'>{max_growth_year}</span> <span class='yoy-positive'>(+{max_growth:.1f}%)</span></div>
                    </div>
                    <div class='yoy-metric-box' style='min-width:160px;'>
                        <div class='yoy-metric-label'>Worst Year</div>
                        <div class='yoy-metric-large'><span class='yoy-year'>{min_growth_year}</span> <span class='yoy-negative'>({min_growth:.1f}%)</span></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Trend analysis
                recent_years = yoy_data.tail(3)
                recent_trend = recent_years['YoY_Change_Pct'].mean()

                # Use consistent styling for recent trend box matching other metric boxes
                trend_class = 'yoy-positive' if recent_trend >= 0 else 'yoy-negative'
                trend_sign = '+' if recent_trend >= 0 else ''
                trend_icon = '📈' if recent_trend >= 0 else '📉'
                trend_text = 'Growing' if recent_trend >= 0 else 'Declining'
                
                st.markdown(f"""
                <div class='yoy-metric-box' style='margin-top:10px;'>
                    <div class='yoy-metric-label'>{trend_icon} Recent 3-year trend</div>
                    <div class='yoy-metric-large'>
                        <span class='{trend_class}'>{trend_sign}{recent_trend:.1f}%</span>
                        <span style='color:#cccccc !important; margin-left:8px; font-size:0.8rem;'>({trend_text})</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Not enough data to calculate year-over-year changes. Need at least 2 years of data.")
    else:
        st.warning("No data available for the selected filters.")
    
    # AI Assistant Section
    st.subheader("🤖 AI Assistant - Ask Questions About the Data")
    
    # AI question input
    user_question = st.text_input(
        "Ask me anything about the crude oil trade data:",
        placeholder="e.g., Which country has the highest oil exports? What was the trend in 2008? Compare imports vs exports for Asia...",
        key="ai_input"
    )
    
    if user_question:
        with st.spinner("🧠 Analyzing your question..."):
            try:
                # Use the original unfiltered dataframe for AI analysis
                ai_response = get_ai_response(user_question, df)
                
                st.markdown("### 💬 Answer:")
                st.markdown(f"**Question:** {user_question}")
                st.markdown(f"<div class='ai-response'>{ai_response}</div>", unsafe_allow_html=True)
                
                # Clear History Button after answer
                st.markdown("---")  # Add a separator line
                col1, col2, col3 = st.columns([2, 1, 2])
                with col2:
                    if st.button("🗑️ Clear Session", key="clear_session", help="Clear current session data"):
                        # Clear the AI input and any session data
                        if 'ai_input' in st.session_state:
                            del st.session_state.ai_input
                        # Clear any other session data that might accumulate
                        for key in list(st.session_state.keys()):
                            if key.startswith('ai_') or key in ['clear_session']:
                                del st.session_state[key]
                        st.rerun()
                
                # Clear the input field after answering
                if 'ai_input' in st.session_state:
                    del st.session_state.ai_input
                
            except Exception as e:
                st.error(f"❌ AI Error: {str(e)}")
                st.info("💡 **Try rephrasing your question or ask about:**")
                st.markdown("""
                - Countries and their trade values
                - Trends over specific years  
                - Comparisons between regions or actions
                - Top performers in imports/exports
                """)
    
    # Footer
    st.markdown("**Data Source:** CrudeOilTrade Database | **Dashboard Created:** October 2025")

# --- AI Response Function ---
def get_ai_response(question, dataframe):
    """
    Generate AI response for natural language questions about the dataframe
    """
    # Use OpenAI if available and enabled
    if AI_ENABLED and client:
        try:
            return get_openai_response(question, dataframe)
        except Exception as e:
            st.warning(f"OpenAI failed ({str(e)}), using fallback analysis...")
            # Fall through to rule-based analysis
    
    # Use rule-based analysis as fallback
    try:
        return analyze_question_advanced(question, dataframe)
    except Exception as e:
        return analyze_question_simple(question, dataframe)

def get_openai_response(question, dataframe):
    """
    Generate AI response using OpenAI API with actual data analysis
    """
    question_lower = question.lower()
    
    # Analyze the question to extract relevant data
    relevant_data = ""
    
    # Extract years if mentioned in question
    years_mentioned = []
    import re
    year_matches = re.findall(r'\b(19|20)\d{2}\b', question)
    if year_matches:
        years_mentioned = [int(match[0] + match[1:]) for match in year_matches]
        filtered_df = dataframe[dataframe['Year'].isin(years_mentioned)]
        if not filtered_df.empty:
            total_value = filtered_df['TradeValue'].sum()
            relevant_data += f"\n• Trade value for years {years_mentioned}: ${total_value/1e12:.2f}T USD"
            relevant_data += f"\n• Records for these years: {len(filtered_df):,}"
            
            # Top countries for these years
            top_countries = filtered_df.groupby('Country')['TradeValue'].sum().nlargest(5)
            relevant_data += f"\n• Top countries: {', '.join([f'{country} (${value/1e9:.1f}B)' for country, value in top_countries.items()])}"
    
    # Check for specific countries mentioned
    country_keywords = ['china', 'usa', 'russia', 'saudi', 'india', 'japan', 'germany', 'uk', 'france']
    for keyword in country_keywords:
        if keyword in question_lower:
            country_data = dataframe[dataframe['Country'].str.contains(keyword, case=False, na=False)]
            if not country_data.empty:
                country_total = country_data['TradeValue'].sum()
                relevant_data += f"\n• {keyword.title()} total trade: ${country_total/1e12:.3f}T USD ({len(country_data):,} records)"
    
    # General dataset context
    total_records = len(dataframe)
    years = sorted(dataframe['Year'].unique())
    total_trade = dataframe['TradeValue'].sum()
    
    context = f"""You are an expert data analyst. Answer the specific question using this crude oil trade data:

RELEVANT DATA FOR THIS QUESTION:{relevant_data}

DATASET OVERVIEW:
- Total Records: {total_records:,}
- Years: {years[0]}-{years[-1]}
- Total Trade Value: ${total_trade/1e12:.2f}T USD

Answer the specific question with the actual data provided above. Be precise and use the exact numbers shown."""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": question}
            ],
            max_tokens=500,
            temperature=0.1
        )
        
        if response and response.choices:
            return response.choices[0].message.content.strip()
        else:
            raise Exception("Empty response from OpenAI")
            
    except Exception as e:
        raise Exception(f"OpenAI API error: {str(e)}")

def analyze_question_advanced(question, df):
    """
    Advanced analysis using pandas operations and natural language processing
    """
    question_lower = question.lower()
    
    # Extract entities and intent from the question
    countries = extract_country_names(question, df)
    years = extract_years(question, df)
    continents = extract_continent_names(question, df)
    
    # Advanced pattern matching with context
    if any(word in question_lower for word in ['compare', 'comparison', 'vs', 'versus', 'difference']):
        return handle_comparison_question(question_lower, df, countries, continents)
    
    elif any(word in question_lower for word in ['trend', 'over time', 'change', 'growth', 'decline']):
        return handle_trend_question(question_lower, df, years, countries)
    
    elif any(word in question_lower for word in ['top', 'highest', 'most', 'best', 'largest', 'maximum']):
        return handle_ranking_question(question_lower, df, countries)
    
    elif any(word in question_lower for word in ['total', 'sum', 'how much', 'amount']):
        return handle_aggregation_question(question_lower, df, countries, years)
    
    elif any(word in question_lower for word in ['when', 'which year', 'what year']):
        return handle_temporal_question(question_lower, df)
    
    elif countries:
        return handle_country_specific_question(question_lower, df, countries)
    
    else:
        # Fallback to simple analysis
        return analyze_question_simple(question, df)

def extract_country_names(question, df):
    """Extract country names mentioned in the question"""
    question_lower = question.lower()
    countries = []
    
    for country in df['Country'].unique():
        if country.lower() in question_lower:
            countries.append(country)
    
    return countries

def extract_years(question, df):
    """Extract years mentioned in the question"""
    import re
    years = []
    
    # Find 4-digit years
    year_matches = re.findall(r'\b(19\d{2}|20\d{2})\b', question)
    for year in year_matches:
        if int(year) in df['Year'].values:
            years.append(int(year))
    
    return years

def extract_continent_names(question, df):
    """Extract continent names mentioned in the question"""
    question_lower = question.lower()
    continents = []
    
    continent_mapping = {
        'europe': 'Europe', 'european': 'Europe',
        'asia': 'Asia', 'asian': 'Asia',
        'africa': 'Africa', 'african': 'Africa',
        'north america': 'North America', 'america': 'North America',
        'south america': 'South America',
        'oceania': 'Oceania'
    }
    
    for key, continent in continent_mapping.items():
        if key in question_lower and continent in df['Continent'].values:
            continents.append(continent)
    
    return continents

def format_currency(value, is_year=False):
    """Format currency values with appropriate suffixes, except for years"""
    if is_year or (1900 <= value <= 2100):
        # Don't format years - keep them as regular numbers
        return str(int(value))
    elif value >= 1e12:
        return f"${value/1e12:.2f}T"
    elif value >= 1e9:
        return f"${value/1e9:.2f}B"
    elif value >= 1e6:
        return f"${value/1e6:.2f}M"
    elif value >= 1e3:
        return f"${value/1e3:.2f}K"
    else:
        return f"${value:.2f}"

def handle_comparison_question(question, df, countries, continents):
    """Handle comparison-type questions"""
    if continents and len(continents) >= 2:
        # Compare specific continents
        comparison_data = {}
        for continent in continents:
            continent_total = df[df['Continent'] == continent]['TradeValue'].sum()
            comparison_data[continent] = continent_total
        
        response = f"**Comparison between {' vs '.join(continents)}:**\n\n"
        for continent, value in comparison_data.items():
            response += f"• **{continent}**: {format_currency(value)}\n"
        
        # Add insights
        highest = max(comparison_data, key=comparison_data.get)
        lowest = min(comparison_data, key=comparison_data.get)
        response += f"\n**Insights:**\n"
        response += f"• {highest} leads with {format_currency(comparison_data[highest])}\n"
        response += f"• {lowest} has {format_currency(comparison_data[lowest])}\n"
        
        return response
    
    elif 'import' in question and 'export' in question:
        # Import vs Export comparison
        import_total = df[df['Action'] == 'Import']['TradeValue'].sum()
        export_total = df[df['Action'] == 'Export']['TradeValue'].sum()
        
        response = "**Global Import vs Export Comparison:**\n\n"
        response += f"• **Total Imports**: {format_currency(import_total)} ({import_total/(import_total + export_total)*100:.1f}%)\n"
        response += f"• **Total Exports**: {format_currency(export_total)} ({export_total/(import_total + export_total)*100:.1f}%)\n"
        
        trade_balance = export_total - import_total
        balance_sign = "+" if trade_balance >= 0 else ""
        response += f"• **Trade Balance**: {balance_sign}{format_currency(abs(trade_balance))}\n"
        
        if export_total > import_total:
            response += f"• **Status**: Global trade surplus\n"
        else:
            response += f"• **Status**: Global trade deficit\n"
        
        return response
    
    else:
        # General continent comparison
        continent_totals = df.groupby('Continent')['TradeValue'].sum().sort_values(ascending=False)
        response = "**Trade Values by Continent:**\n\n"
        for continent, value in continent_totals.head(6).items():
            response += f"• **{continent}**: {format_currency(value)}\n"
        
        return response

def handle_trend_question(question, df, years, countries):
    """Handle trend-related questions"""
    if years and len(years) >= 2:
        # Specific year range trend
        start_year, end_year = min(years), max(years)
        trend_data = df[df['Year'].between(start_year, end_year)].groupby('Year')['TradeValue'].sum()
    elif countries:
        # Country-specific trend
        country_data = df[df['Country'].isin(countries)]
        trend_data = country_data.groupby('Year')['TradeValue'].sum()
        
        response = f"**Trade Trend for {', '.join(countries)}:**\n\n"
        start_value = trend_data.iloc[0]
        end_value = trend_data.iloc[-1]
        peak_year = trend_data.idxmax()
        peak_value = trend_data.max()
        
        response += f"• **Period**: {trend_data.index.min()}-{trend_data.index.max()}\n"
        response += f"• **Starting Value**: {format_currency(start_value)}\n"
        response += f"• **Ending Value**: {format_currency(end_value)}\n"
        response += f"• **Peak Year**: {peak_year} ({format_currency(peak_value)})\n"
        response += f"• **Overall Change**: {((end_value - start_value) / start_value * 100):+.1f}%\n"
        
        return response
    else:
        # Global trend
        trend_data = df.groupby('Year')['TradeValue'].sum()
    
    start_value = trend_data.iloc[0]
    end_value = trend_data.iloc[-1]
    peak_year = trend_data.idxmax()
    peak_value = trend_data.max()
    
    response = "**Global Trade Trend Analysis:**\n\n"
    response += f"• **Period**: {trend_data.index.min()}-{trend_data.index.max()}\n"
    response += f"• **Starting Value**: {format_currency(start_value)}\n"
    response += f"• **Ending Value**: {format_currency(end_value)}\n"
    response += f"• **Peak Year**: {peak_year} ({format_currency(peak_value)})\n"
    response += f"• **Overall Growth**: {((end_value - start_value) / start_value * 100):+.1f}%\n"
    
    # Calculate average annual growth
    years_span = trend_data.index.max() - trend_data.index.min()
    cagr = ((end_value / start_value) ** (1/years_span) - 1) * 100
    response += f"• **Average Annual Growth**: {cagr:+.1f}%\n"
    
    return response

def handle_ranking_question(question, df, countries):
    """Handle ranking/top-related questions"""
    if 'import' in question:
        ranking_data = df[df['Action'] == 'Import'].groupby('Country')['TradeValue'].sum().nlargest(10)
        response = "**Top 10 Oil Importing Countries:**\n\n"
    elif 'export' in question:
        ranking_data = df[df['Action'] == 'Export'].groupby('Country')['TradeValue'].sum().nlargest(10)
        response = "**Top 10 Oil Exporting Countries:**\n\n"
    else:
        ranking_data = df.groupby('Country')['TradeValue'].sum().nlargest(10)
        response = "**Top 10 Countries by Total Trade Value:**\n\n"
    
    for i, (country, value) in enumerate(ranking_data.items(), 1):
        response += f"{i}. **{country}**: {format_currency(value)}\n"
    
    # Add market share info
    total_in_category = ranking_data.sum()
    overall_total = df['TradeValue'].sum() if 'import' not in question and 'export' not in question else df[df['Action'] == ('Import' if 'import' in question else 'Export')]['TradeValue'].sum()
    
    response += f"\n**Top 10 Market Share**: {(total_in_category/overall_total*100):.1f}%\n"
    
    return response

def handle_aggregation_question(question, df, countries, years):
    """Handle total/sum/amount questions"""
    filtered_df = df.copy()
    
    if countries:
        filtered_df = filtered_df[filtered_df['Country'].isin(countries)]
    if years:
        filtered_df = filtered_df[filtered_df['Year'].isin(years)]
    
    if 'import' in question:
        total = filtered_df[filtered_df['Action'] == 'Import']['TradeValue'].sum()
        action_text = "Imports"
    elif 'export' in question:
        total = filtered_df[filtered_df['Action'] == 'Export']['TradeValue'].sum()
        action_text = "Exports"
    else:
        total = filtered_df['TradeValue'].sum()
        action_text = "Trade"
    
    context = []
    if countries:
        context.append(f"Countries: {', '.join(countries)}")
    if years:
        context.append(f"Years: {', '.join(map(str, years))}")
    
    response = f"**Total {action_text}"
    if context:
        response += f" ({' | '.join(context)})"
    response += ":**\n\n"
    
    response += f"• **Amount**: {format_currency(total)}\n"
    
    # Add percentage of global total
    global_total = df['TradeValue'].sum()
    response += f"• **Global Share**: {(total/global_total*100):.1f}%\n"
    
    return response

def handle_temporal_question(question, df):
    """Handle when/which year questions"""
    if any(word in question for word in ['highest', 'peak', 'maximum', 'most']):
        yearly_totals = df.groupby('Year')['TradeValue'].sum()
        peak_year = yearly_totals.idxmax()
        peak_value = yearly_totals.max()
        
        response = f"**Peak Trade Year:**\n\n"
        response += f"• **Year**: {peak_year}\n"
        response += f"• **Trade Value**: {format_currency(peak_value)}\n"
        
        # Context about what happened
        year_data = df[df['Year'] == peak_year]
        top_trader = year_data.groupby('Country')['TradeValue'].sum().idxmax()
        response += f"• **Top Trading Country**: {top_trader}\n"
        
        return response
    
    elif any(word in question for word in ['lowest', 'minimum', 'least']):
        yearly_totals = df.groupby('Year')['TradeValue'].sum()
        low_year = yearly_totals.idxmin()
        low_value = yearly_totals.min()
        
        response = f"**Lowest Trade Year:**\n\n"
        response += f"• **Year**: {low_year}\n"
        response += f"• **Trade Value**: {format_currency(low_value)}\n"
        
        return response
    
    return analyze_question_simple(question, df)

def handle_country_specific_question(question, df, countries):
    """Handle questions about specific countries"""
    country_data = df[df['Country'].isin(countries)]
    
    response = f"**Analysis for {', '.join(countries)}:**\n\n"
    
    for country in countries:
        country_specific = df[df['Country'] == country]
        total_trade = country_specific['TradeValue'].sum()
        
        imports = country_specific[country_specific['Action'] == 'Import']['TradeValue'].sum()
        exports = country_specific[country_specific['Action'] == 'Export']['TradeValue'].sum()
        
        response += f"**{country}:**\n"
        response += f"• Total Trade: {format_currency(total_trade)}\n"
        response += f"• Imports: {format_currency(imports)}\n"
        response += f"• Exports: {format_currency(exports)}\n"
        trade_balance = exports - imports
        balance_sign = "+" if trade_balance >= 0 else ""
        response += f"• Trade Balance: {balance_sign}{format_currency(abs(trade_balance))}\n\n"
    
    return response

def analyze_question_simple(question, df):
    """
    Simple rule-based analysis for common questions about the dataframe
    """
    question_lower = question.lower()
    
    # Question type detection
    if any(word in question_lower for word in ['highest', 'most', 'top', 'largest', 'maximum']):
        if 'import' in question_lower:
            # Top importers
            top_importers = df[df['Action'] == 'Import'].groupby('Country')['TradeValue'].sum().nlargest(5)
            response = "**Top 5 Oil Importing Countries:**\n\n"
            for i, (country, value) in enumerate(top_importers.items(), 1):
                response += f"{i}. **{country}**: {format_currency(value)}\n"
            return response
            
        elif 'export' in question_lower:
            # Top exporters
            top_exporters = df[df['Action'] == 'Export'].groupby('Country')['TradeValue'].sum().nlargest(5)
            response = "**Top 5 Oil Exporting Countries:**\n\n"
            for i, (country, value) in enumerate(top_exporters.items(), 1):
                response += f"{i}. **{country}**: {format_currency(value)}\n"
            return response
            
        elif any(word in question_lower for word in ['trade', 'value', 'total']):
            # Highest trade values
            top_countries = df.groupby('Country')['TradeValue'].sum().nlargest(5)
            response = "**Top 5 Countries by Total Trade Value:**\n\n"
            for i, (country, value) in enumerate(top_countries.items(), 1):
                response += f"{i}. **{country}**: {format_currency(value)}\n"
            return response
    
    elif any(word in question_lower for word in ['trend', 'over time', 'years', 'change']):
        # Trend analysis
        yearly_totals = df.groupby('Year')['TradeValue'].sum().sort_index()
        start_value = yearly_totals.iloc[0]
        end_value = yearly_totals.iloc[-1]
        peak_year = yearly_totals.idxmax()
        peak_value = yearly_totals.max()
        
        response = f"**Global Crude Oil Trade Trends (1995-2021):**\n\n"
        response += f"• **Starting Value (1995)**: {format_currency(start_value)}\n"
        response += f"• **Ending Value (2021)**: {format_currency(end_value)}\n"
        response += f"• **Peak Year**: {peak_year} ({format_currency(peak_value)})\n"
        response += f"• **Overall Growth**: {((end_value - start_value) / start_value * 100):+.1f}%\n"
        
        return response
    
    elif 'compare' in question_lower or 'vs' in question_lower or 'versus' in question_lower:
        # Comparison analysis
        if any(continent in question_lower for continent in ['north america', 'europe', 'asia', 'africa', 'south america', 'oceania']):
            continent_totals = df.groupby('Continent')['TradeValue'].sum().sort_values(ascending=False)
            response = "**Trade Values by Continent:**\n\n"
            for continent, value in continent_totals.items():
                response += f"• **{continent}**: {format_currency(value)}\n"
            return response
            
        elif 'import' in question_lower and 'export' in question_lower:
            # Import vs Export comparison
            import_total = df[df['Action'] == 'Import']['TradeValue'].sum()
            export_total = df[df['Action'] == 'Export']['TradeValue'].sum()
            
            response = "**Global Import vs Export Comparison:**\n\n"
            response += f"• **Total Imports**: {format_currency(import_total)}\n"
            response += f"• **Total Exports**: {format_currency(export_total)}\n"
            
            trade_balance = export_total - import_total
            balance_sign = "+" if trade_balance >= 0 else ""
            response += f"• **Trade Balance**: {balance_sign}{format_currency(abs(trade_balance))}\n"
            response += f"• **Import Share**: {import_total/(import_total + export_total)*100:.1f}%\n"
            response += f"• **Export Share**: {export_total/(import_total + export_total)*100:.1f}%\n"
            
            return response
    
    elif any(word in question_lower for word in ['when', 'which year', 'what year']):
        if any(word in question_lower for word in ['highest', 'peak', 'maximum']):
            yearly_totals = df.groupby('Year')['TradeValue'].sum()
            peak_year = yearly_totals.idxmax()
            peak_value = yearly_totals.max()
            
            response = f"**Peak Trade Year:**\n\n"
            response += f"• **Year**: {peak_year}\n"
            response += f"• **Trade Value**: {format_currency(peak_value)}\n"
            
            # Context about what happened that year
            year_data = df[df['Year'] == peak_year]
            top_country = year_data.groupby('Country')['TradeValue'].sum().idxmax()
            response += f"• **Top Trading Country**: {top_country}\n"
            
            return response
    
    elif any(word in question_lower for word in ['country', 'countries']):
        # Country-specific analysis
        total_countries = df['Country'].nunique()
        top_country = df.groupby('Country')['TradeValue'].sum().idxmax()
        top_value = df.groupby('Country')['TradeValue'].sum().max()
        
        response = f"**Country Statistics:**\n\n"
        response += f"• **Total Countries**: {total_countries}\n"
        response += f"• **Top Trading Country**: {top_country}\n"
        response += f"• **Top Country Value**: {format_currency(top_value)}\n"
        
        return response
    
    else:
        # General data overview
        total_trade = df['TradeValue'].sum()
        total_records = len(df)
        year_range = f"{df['Year'].min()}-{df['Year'].max()}"
        
        response = f"**Dataset Overview:**\n\n"
        response += f"• **Total Trade Value**: {format_currency(total_trade)}\n"
        response += f"• **Total Records**: {total_records:,}\n"
        response += f"• **Year Range**: {year_range}\n"
        response += f"• **Countries Covered**: {df['Country'].nunique()}\n"
        response += f"• **Continents**: {df['Continent'].nunique()}\n"
        
        response += f"\n*Ask me specific questions about trends, comparisons, or top performers!*"
        
        return response

# --- Original Analysis Function (for terminal usage) ---
def run_terminal_analysis():
    """Run the original terminal-based analysis"""
    print("🌍 Crude Oil Trade Analysis")
    print("=" * 40)
    
    if error:
        print(f"❌ Database connection failed: {error}")
        return
    
    if df.empty:
        print("⚠️ No data available from the database.")
        return
    
    print("✅ Connection successful!")
    print(f"✅ Data retrieved successfully! ({len(df)} records)")
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Analysis
    total_records = len(df)
    average_trade_value = df['TradeValue'].mean()
    
    print(f"\n📊 Crude Oil Data Analysis")
    print(f"Total Records: {total_records:,}")
    print(f"Average Trade Value: ${average_trade_value:,.2f}")
    
    print(f"\nTop 5 Countries by Trade Value:")
    top_countries = df.nlargest(5, 'TradeValue')[['Country', 'TradeValue', 'Action']]
    for _, row in top_countries.iterrows():
        print(f"  {row['Country']}: ${row['TradeValue']:,.2f} ({row['Action']})")

# --- Main Execution ---
if __name__ == "__main__":
    import sys
    
    # Check if running in Streamlit
    if 'streamlit' in sys.modules:
        main()
    else:
        # Running in terminal - show original analysis
        run_terminal_analysis()


