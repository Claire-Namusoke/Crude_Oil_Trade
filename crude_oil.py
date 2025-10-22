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

# AI imports
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    st.warning("⚠️ AI libraries not installed. AI features will use fallback analysis.")

# Set up OpenAI client if available
if OPENAI_AVAILABLE:
    # Read API key from Streamlit Cloud secrets
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        if api_key and len(api_key) > 0:
            client = OpenAI(api_key=api_key)
            AI_ENABLED = True
        else:
            client = None
            AI_ENABLED = False
            st.error("🔑 OpenAI API key is empty in Streamlit secrets")
    except KeyError:
        # Key not found in secrets
        client = None
        AI_ENABLED = False
        st.error("🔑 OPENAI_API_KEY not found in Streamlit Cloud secrets. Please add it in Settings → Secrets")
    except Exception as e:
        # Other initialization errors
        client = None
        AI_ENABLED = False
        st.error(f"❌ OpenAI initialization failed: {str(e)}")
else:
    client = None
    AI_ENABLED = False
    st.error("❌ OpenAI library not installed")

# --- Utility Functions ---

# Regional mapping for African countries
region_map = {
    "west africa": ["Nigeria", "Ghana", "Senegal", "Côte d'Ivoire", "Sierra Leone", "Liberia", "Togo", "Benin", "Gambia", "Cape Verde", "Guinea", "Guinea-Bissau"],
    "east africa": ["Kenya", "Uganda", "Tanzania", "Ethiopia", "Somalia", "Rwanda", "Burundi", "Sudan", "Eritrea", "Djibouti", "South Sudan"],
    "north africa": ["Egypt", "Libya", "Algeria", "Morocco", "Tunisia", "Mauritania"],
    "southern africa": ["South Africa", "Namibia", "Botswana", "Zimbabwe", "Zambia", "Lesotho", "Eswatini", "Malawi", "Angola"],
    "central africa": ["Cameroon", "Congo", "Gabon", "Chad", "Equatorial Guinea", "Central African Republic", "Democratic Republic of the Congo"],
}

def format_human_readable(value):
    """Convert large numbers into human-readable format: K/M/B/T"""
    if pd.isna(value) or value == 0:
        return "$0"
    
    abs_value = abs(value)
    sign = "-" if value < 0 else ""
    
    if abs_value >= 1e12:
        return f"{sign}${abs_value/1e12:.2f}T"
    elif abs_value >= 1e9:
        return f"{sign}${abs_value/1e9:.2f}B"
    elif abs_value >= 1e6:
        return f"{sign}${abs_value/1e6:.2f}M"
    elif abs_value >= 1e3:
        return f"{sign}${abs_value/1e3:.2f}K"
    else:
        return f"{sign}${abs_value:.2f}"

system_prompt = """
You are a data reasoning assistant specializing in crude oil trade data. 
You can interpret complex questions about countries, continents, years, or time ranges.

Your goal:
- Understand exactly what the user is asking, even if phrased in different natural language forms.
- Always filter and reason strictly using the dataset provided — never guess or make up data.
- Identify key question entities such as:
  - Specific country names (e.g., Nigeria, China, United States)
  - Continents (e.g., Africa, Europe, Asia)
  - Year or range (e.g., 2000, between 2000 and 2008)
  - Import or Export context

Rules for answering:
1. Filter the dataset based on what the question asks for (continent/country/year/import/export).
2. If the question says "between [year1] and [year2]", analyze and summarize across that range.
3. If the question says "which country imported less in Africa", compare *only African countries* for that year.
4. If the user doesn’t specify, clarify assumptions explicitly (e.g., "Assuming you mean total exports for Africa in 2000...").
5. Return **only clear human-readable results**, like:
   Nigeria in 2000: $22.4B exports.
   South Africa in 2000: $4.8B imports.
6. Always use the correct unit (K, M, B, or T) — the dataset already includes trade values that should be scaled using the format_human_readable() function.
7. Never include markdown symbols like **, *, or backticks — output should be clean and plain text.
8. When comparing, show concise context:
   “Trade decreased by 11.1% between 2000 and 2001” (not long paragraphs).

Be direct, precise, and intelligent in reasoning — think like a professional data analyst who fully understands the dataset.
"""

# ------------------------
# Data Analysis Functions - These do the actual DataFrame work
# ------------------------
def analyze_lowest_highest(df, metric_type, action_type=None):
    """Find country with lowest/highest trade value"""
    df_work = df.copy()
    if action_type:
        df_work = df_work[df_work['Action'] == action_type]
    
    if df_work.empty:
        return None
    
    country_totals = df_work.groupby('Country')['TradeValue'].sum().sort_values()
    
    if metric_type == "lowest":
        country = country_totals.index[0]
        value = country_totals.iloc[0]
    else:  # highest
        country = country_totals.index[-1]
        value = country_totals.iloc[-1]
    
    return {
        "country": country,
        "value": value,
        "value_formatted": format_human_readable(value),
        "action": action_type or "Trade",
        "metric": metric_type
    }

def analyze_country_specific(df, country_name, years=None, action=None):
    """Get specific country data"""
    df_work = df[df['Country'].str.lower() == country_name.lower()].copy()
    
    if years:
        df_work = df_work[df_work['Year'].isin(years)]
    if action:
        df_work = df_work[df_work['Action'] == action]
    
    if df_work.empty:
        return None
    
    total = df_work['TradeValue'].sum()
    return {
        "country": country_name,
        "total": total,
        "total_formatted": format_human_readable(total),
        "years": years,
        "action": action
    }

def analyze_year_range(df, start_year, end_year):
    """Analyze trends across year range"""
    df_work = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)].copy()
    
    if df_work.empty:
        return None
    
    yearly_totals = df_work.groupby('Year')['TradeValue'].sum().sort_index()
    start_val = yearly_totals.iloc[0]
    end_val = yearly_totals.iloc[-1]
    change = end_val - start_val
    pct_change = (change / start_val * 100) if start_val != 0 else 0
    
    return {
        "start_year": start_year,
        "end_year": end_year,
        "start_value": start_val,
        "end_value": end_val,
        "start_formatted": format_human_readable(start_val),
        "end_formatted": format_human_readable(end_val),
        "change": change,
        "change_formatted": format_human_readable(abs(change)),
        "pct_change": pct_change
    }

# ------------------------
# Main Answer Function with Real Data Analysis
# ------------------------
def answer_question(user_question, df):
    """
    Analyzes the actual DataFrame to get real results,
    then uses AI to format a natural language response.
    """
    question_lower = user_question.lower()
    
    # Extract years from question
    years = [int(token) for token in user_question.split() if token.isdigit() and len(token) == 4]
    
    # Extract continent filter
    continents = ["Africa", "Europe", "Asia", "North America", "South America", "Oceania"]
    continent = None
    for cont in continents:
        if cont.lower() in question_lower:
            continent = cont
    
    # Apply filters
    df_filtered = df.copy()
    if continent:
        df_filtered = df_filtered[df_filtered["Continent"] == continent]
    if years:
        df_filtered = df_filtered[df_filtered["Year"].isin(years)]
    
    # Perform actual data analysis based on question type
    analysis_result = None
    
    # Lowest/Highest queries
    if "lowest" in question_lower:
        if "import" in question_lower:
            analysis_result = analyze_lowest_highest(df_filtered, "lowest", "Import")
        elif "export" in question_lower:
            analysis_result = analyze_lowest_highest(df_filtered, "lowest", "Export")
        else:
            # Generic lowest trade
            analysis_result = analyze_lowest_highest(df_filtered, "lowest")
    
    elif "highest" in question_lower:
        if "import" in question_lower:
            analysis_result = analyze_lowest_highest(df_filtered, "highest", "Import")
        elif "export" in question_lower:
            analysis_result = analyze_lowest_highest(df_filtered, "highest", "Export")
        else:
            # Generic highest trade
            analysis_result = analyze_lowest_highest(df_filtered, "highest")
    
    # Year range analysis
    elif len(years) >= 2 and any(word in question_lower for word in ["between", "from", "compare", "trend", "change"]):
        analysis_result = analyze_year_range(df_filtered, min(years), max(years))
    
    # Country-specific queries
    elif any(country.lower() in question_lower for country in df['Country'].unique()):
        # Find the country mentioned
        for country in df['Country'].unique():
            if country.lower() in question_lower:
                action = "Export" if "export" in question_lower else "Import" if "import" in question_lower else None
                analysis_result = analyze_country_specific(df_filtered, country, years, action)
                break
    
    # Total/Summary queries
    else:
        total = df_filtered["TradeValue"].sum()
        countries = df_filtered["Country"].nunique()
        analysis_result = {
            "type": "summary",
            "total": total,
            "total_formatted": format_human_readable(total),
            "countries": countries,
            "years": years if years else "all years"
        }

    # ------------------------
    # Send REAL analysis result to AI for natural-language formatting
    # ------------------------
    if analysis_result is None:
        return "I couldn't find relevant data to answer that question. Please try rephrasing or ask about specific countries, years, or trade metrics."
    
    if AI_ENABLED and client:
        # Create a clear prompt with the actual data analysis
        prompt = f"""You are a data analyst. Format this crude oil trade analysis result into a clear, concise answer.

User Question: {user_question}

ACTUAL DATA ANALYSIS RESULT:
{analysis_result}

Instructions:
- Answer directly and naturally
- Use the exact numbers from the analysis result
- Be specific (mention country names, values, years)
- Keep it concise (1-2 sentences)
- Don't add extra information not in the data
- Don't use markdown formatting

Example: "Russia had the highest exports with $523.4B in total trade value."
"""

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            # Fallback to formatted result if AI fails
            return f"Analysis complete: {analysis_result}"
    else:
        # Format result without AI
        if "country" in analysis_result:
            country = analysis_result["country"]
            value = analysis_result.get("value_formatted") or analysis_result.get("total_formatted", "")
            action = analysis_result.get("action", "trade")
            metric = analysis_result.get("metric", "")
            return f"{country} has the {metric} {action} value with {value}"
        else:
            return f"Analysis result: {analysis_result}"


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
        # Test with the new OpenAI client format
        test_client = OpenAI(api_key=api_key)
        
        # Test with a very simple completion
        response = test_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello"}
            ],
            max_tokens=10
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
    continent_options = ['All'] + sorted(df['Continent'].unique())
    selected_continents = st.sidebar.multiselect(
        "🌍 Select Continents:",
        options=continent_options,
        default=['All']
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
        
    if selected_continents and 'All' not in selected_continents:
        filtered_df = filtered_df[filtered_df['Continent'].isin(selected_continents)]
    
    # For display purposes, determine if showing all continents
    showing_all_continents = not selected_continents or 'All' in selected_continents
    selected_continent = 'All' if showing_all_continents else selected_continents[0] if len(selected_continents) == 1 else 'Multiple'
    
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
        
        # Format trade value using human-readable format
        trade_value_formatted = format_human_readable(total_trade)
            
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
            import_formatted = format_human_readable(import_total)
            export_formatted = format_human_readable(export_total)
            
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
            
            balance_formatted = format_human_readable(abs(trade_balance))
            
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
            
            st.dataframe(styled_importers, use_container_width=True)
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
            
            st.dataframe(styled_exporters, use_container_width=True)
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
            'Congo': 'COG', 'Democratic Republic of the Congo': 'COD', 'Sudan': 'SDN',
            # --- AUTO-ADDED MISSING ISO CODES ---
            'Papua New Guinea': 'PNG', 'Fiji': 'FJI', 'Solomon Islands': 'SLB', 'New Caledonia': 'NCL',
            'French Polynesia': 'PYF', 'Vanuatu': 'VUT', 'Samoa': 'WSM', 'Kiribati': 'KIR',
            'Federated States of Micronesia': 'FSM', 'Tonga': 'TON', 'Marshall Islands': 'MHL', 'Palau': 'PLW',
            'Cook Islands': 'COK', 'Nauru': 'NRU', 'Tuvalu': 'TUV', 'Niue': 'NIU', 'American Samoa': 'ASM',
            'Guam': 'GUM', 'Northern Mariana Islands': 'MNP', 'Tokelau': 'TKL', 'Wallis and Futuna': 'WLF'
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
    
    # Initialize session state for clearing input
    if 'clear_input' not in st.session_state:
        st.session_state.clear_input = False
    
    # Clear the input if flag is set
    if st.session_state.clear_input:
        st.session_state.ai_input = ""
        st.session_state.clear_input = False
    
    # AI question input
    user_question = st.text_input(
        "Ask me anything about the crude oil trade data:",
        placeholder="e.g., Which country has the highest oil exports? What was the trend in 2008? Compare imports vs exports for Asia...",
        key="ai_input"
    )
    
    if st.button("Get Answer") and user_question:
        with st.spinner("🧠 Analyzing your question..."):
            try:
                # Use the answer_question function for AI analysis
                ai_response = answer_question(user_question, df)
                
                st.markdown("### 💬 Answer:")
                st.markdown(f"**Question:** {user_question}")
                st.markdown(f"<div class='ai-response'>{ai_response}</div>", unsafe_allow_html=True)
                
                # Clear Session Button after answer
                st.markdown("---")
                col1, col2, col3 = st.columns([2, 1, 2])
                with col2:
                    if st.button("🗑️ Clear Session", key="clear_session", help="Clear question and start fresh"):
                        st.session_state.clear_input = True
                        st.rerun()
                
                # Set flag to clear input on next rerun
                st.session_state.clear_input = True
                
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
    Generate AI response using LangChain pandas agent (much more reliable!)
    """
    # Use Haystack if available
    # Use the new answer_question function
    if AI_ENABLED:
        try:
            return answer_question(question, dataframe)
        except Exception as e:
            st.warning(f"AI analysis failed ({str(e)}), using basic analysis...")
            return get_basic_response(question, dataframe)
    
    # Use basic pandas analysis as fallback
    try:
        return get_basic_response(question, dataframe)
    except Exception as e:
        return f"Sorry, I couldn't analyze that question: {str(e)}"

def get_haystack_response(question, dataframe):
    """
    Intelligent AI assistant that carefully analyzes user questions and provides comprehensive answers
    """
    try:
        # Use the enhanced intelligent analyzer
        return intelligent_dataframe_analyzer(question, dataframe)
        
    except Exception as e:
        return f"I encountered an error while analyzing your question: {str(e)}. Please try rephrasing your question."

def intelligent_dataframe_analyzer(question, dataframe):
    """
    Expert Data Analyst AI System with Context-Aware Analysis:
    
    Enhanced with full dataset context for improved accuracy.
    Filters DataFrame first, then provides natural-language answers with complete dataset context.
    """
    try:
        # Use context-aware analysis for better accuracy
        if AI_ENABLED:
            return context_aware_analysis(question, dataframe)
        else:
            # Fallback to enhanced rule-based analysis
            return rule_based_intelligent_analysis(question, dataframe)
        
    except Exception as e:
        return f"I encountered an error analyzing your question: {str(e)}. Please try asking in a different way."

def context_aware_analysis(user_question, dataframe):
    """
    Context-aware AI analysis that provides full dataset context for improved accuracy.
    Filters the DataFrame first, then forms natural-language answers with complete context.
    """
    try:
        # Step 1: Pre-filter the dataset based on the question
        filtered_df = pre_filter_dataset(user_question, dataframe)
        
        # Step 2: Convert filtered dataset to context format
        if len(filtered_df) > 100:  # Limit context size for large datasets
            # Sample strategically: recent years + top countries by trade value
            recent_years = filtered_df[filtered_df['Year'] >= 2015]
            top_countries = filtered_df.groupby('Country')['TradeValue'].sum().nlargest(20).index
            top_country_data = filtered_df[filtered_df['Country'].isin(top_countries)]
            context_df = pd.concat([recent_years, top_country_data]).drop_duplicates()
        else:
            context_df = filtered_df
            
        # Convert to dictionary format for context
        context = context_df.to_dict(orient='records')
        
        # Step 3: Create comprehensive system prompt
        system_prompt = """
You are an expert, careful data analyst working only with the dataset provided in the "Dataset context" section. Your job is to interpret the user's natural-language question about the dataset, extract the intent precisely, compute or retrieve the exact answer from the dataset context, and return a short, direct, and accurate response. Do NOT invent facts or use outside knowledge.

Rules (apply these strictly):
1. Always extract explicit entities (country, region, continent), numeric filters (years or ranges), and operation intent (sum/total, compare/difference, trend, top-k, average, min/max, raw lookup).
2. If a country or region name appears in the user's question, prefer an **exact string match** (case-insensitive) to the dataset. If exact match fails, do a fuzzy match only as a fallback and **state** that you used a fuzzy match.
3. When the question uses the word "total", "sum", "overall", or asks "between X and Y" with no comparison word (change/increase/decrease), treat it as a **sum across all years in that inclusive range**.
4. When the question includes words like "change", "difference", "increase", "decrease", "compare", "growth", treat it as a **comparison between the first and last year** (or between explicitly requested years).
5. For ambiguous questions, first try to disambiguate by exact matching and intent detection; if still ambiguous, ask a single concise clarifying question (only if absolutely necessary). Otherwise, choose the most likely interpretation and state your assumption briefly.
6. Use only the values present in the provided dataset context. If the requested data is missing for part of the query, explicitly say which part is missing.
7. Always format numeric currency outputs with the dataset's units (K, M, B, T) as provided in the context, or format them using the rules: K = thousand, M = million, B = billion, T = trillion. Show 2 decimal places for scaled values (e.g., $1.23M USD).
8. When returning results, produce a very short answer (1–3 sentences) that includes: the country/region, the year(s) or period, the number(s) + unit, and a single-sentence interpretation if applicable (e.g., percent change).
9. If multiple rows match a query (e.g., asking for "top exporters in 2000"), return the top N results only (user may specify N; otherwise default to top 5) and state the metric and ordering rule used.
10. If user asks for raw data rows or entire dataframe snippets, return a concise table-formatted snippet (max 10 rows) and say where to fetch the rest.

Dataset context contains fields: Country, Continent, Year, Action (Export/Import), TradeValue (in USD).
"""

        # Step 4: Create the full prompt with context
        prompt = f"""
{system_prompt}

Here is the filtered dataset context (most relevant records for your query):
{context}

User question: {user_question}

Answer (be direct, precise, and follow the rules above):
"""

        # Step 5: Get AI response
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert data analyst. Analyze the provided dataset context and answer the user's question accurately and concisely."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.1
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        # Fallback to rule-based analysis
        return rule_based_intelligent_analysis(user_question, dataframe)

def pre_filter_dataset(question, dataframe):
    """
    Pre-filter the dataset based on the user's question to provide relevant context.
    """
    import re
    
    question_lower = question.lower()
    filtered_df = dataframe.copy()
    
    # Extract entities from question
    countries = extract_countries_from_query(question_lower, dataframe)
    years = [int(year) for year in re.findall(r'\b(?:19|20)\d{2}\b', question_lower)]
    
    # Filter by countries if mentioned
    if countries:
        filtered_df = filtered_df[filtered_df['Country'].isin(countries)]
    
    # Filter by years if mentioned
    if years:
        if len(years) == 1:
            filtered_df = filtered_df[filtered_df['Year'] == years[0]]
        elif len(years) >= 2:
            start_year, end_year = min(years), max(years)
            filtered_df = filtered_df[
                (filtered_df['Year'] >= start_year) & 
                (filtered_df['Year'] <= end_year)
            ]
    
    # Filter by action type if mentioned
    if 'export' in question_lower and 'import' not in question_lower:
        filtered_df = filtered_df[filtered_df['Action'] == 'Export']
    elif 'import' in question_lower and 'export' not in question_lower:
        filtered_df = filtered_df[filtered_df['Action'] == 'Import']
    
    # If no specific filtering was applied, limit to recent years for context efficiency
    if len(countries) == 0 and len(years) == 0:
        # Include last 10 years + top 20 countries by total trade value
        recent_years = filtered_df[filtered_df['Year'] >= (filtered_df['Year'].max() - 9)]
        top_countries = filtered_df.groupby('Country')['TradeValue'].sum().nlargest(20).index
        top_country_data = filtered_df[filtered_df['Country'].isin(top_countries)]
        filtered_df = pd.concat([recent_years, top_country_data]).drop_duplicates()
    
    return filtered_df

def rule_based_intelligent_analysis(question, dataframe):
    """
    Enhanced rule-based analysis system (fallback when AI is not available).
    
    CORE PRINCIPLES:
    1. Extract explicit entities (country, region, continent), numeric filters (years/ranges), and operation intent precisely
    2. Use ONLY dataset values - never invent facts or use outside knowledge
    3. Prefer exact string matches (case-insensitive) for countries/regions; state if using fuzzy matching
    4. Format currency with K/M/B/T units, 2 decimal places (e.g., $1.23B USD)
    5. Return 1-3 sentences maximum with country/region, year(s), number(s) + unit
    
    YEAR RANGE LOGIC:
    6. "total"/"sum"/"overall"/"between X and Y" (no comparison words) → Sum across ALL years in range
    7. "change"/"difference"/"increase"/"decrease"/"compare"/"growth" → Compare FIRST vs LAST year only
    8. For ambiguous queries, choose most likely interpretation and state assumption
    
    DATA INTEGRITY:
    9. Always specify country and years explicitly in response
    10. State clearly if requested data is missing from dataset
    11. For top-N queries, default to top 5 unless specified, show metric and ordering
    12. No global assumptions unless explicitly requested by user
    """
    try:
        # Step 1: Carefully read and parse the user's question
        question_lower = question.lower().strip()
        
        # Step 2: Identify what the question is asking for
        analysis_intent = identify_question_intent(question_lower, dataframe)
        
        if not analysis_intent['valid']:
            return analysis_intent['message']
        
        # Step 3 & 4: Use dataset information and provide clear explanations
        result = execute_intelligent_analysis(analysis_intent, dataframe)
        
        return result
        
    except Exception as e:
        return f"I encountered an error analyzing your question: {str(e)}. Please try asking in a different way."

def identify_question_intent(question_lower, dataframe):
    """
    Step 1 & 2: Carefully analyze the question and identify the intent
    """
    import re
    
    # Extract key components
    countries = extract_countries_from_query(question_lower, dataframe)
    continents = extract_continents_from_query(question_lower, dataframe)
    years = [int(year) for year in re.findall(r'\b(?:19|20)\d{2}\b', question_lower)]
    
    # Enhanced intent detection with operation types
    intent = {
        'valid': True,
        'type': None,
        'countries': countries,
        'continents': continents,
        'years': years,
        'action': None,  # Export, Import, or Both
        'operation': None,  # sum, compare, trend, top-k, average, min-max, lookup
        'comparison': False,
        'ranking': False,
        'specific_data': False,
        'temporal_analysis': False,
        'original_question': question_lower,
        'exact_match': True,  # Track if exact country match was used
        'message': ''
    }
    
    # Determine action type
    if 'export' in question_lower and 'import' not in question_lower:
        intent['action'] = 'Export'
    elif 'import' in question_lower and 'export' not in question_lower:
        intent['action'] = 'Import'
    elif 'export' in question_lower and 'import' in question_lower:
        intent['action'] = 'Both'
    elif 'trade' in question_lower:
        intent['action'] = 'Both'
    else:
        intent['action'] = 'Both'  # Default to both if unclear
    
    # Determine operation type based on keywords
    if any(word in question_lower for word in ['total', 'sum', 'overall', 'combined']) and not any(word in question_lower for word in ['change', 'difference', 'increase', 'decrease']):
        intent['operation'] = 'sum'
    elif any(word in question_lower for word in ['change', 'difference', 'increase', 'decrease', 'compare', 'growth', 'versus', 'vs']):
        intent['operation'] = 'compare'
    elif any(word in question_lower for word in ['trend', 'over time', 'pattern']):
        intent['operation'] = 'trend'
    elif any(word in question_lower for word in ['top', 'highest', 'most', 'bottom', 'lowest', 'least', 'rank']):
        intent['operation'] = 'top-k'
    elif any(word in question_lower for word in ['average', 'mean']):
        intent['operation'] = 'average'
    elif any(word in question_lower for word in ['minimum', 'maximum', 'min', 'max']):
        intent['operation'] = 'min-max'
    else:
        intent['operation'] = 'lookup'  # Default raw data lookup
    
    # Identify question patterns
    if any(word in question_lower for word in ['which country', 'what country', 'highest', 'lowest', 'most', 'least', 'top', 'bottom']):
        intent['type'] = 'ranking'
        intent['ranking'] = True
        
    elif any(word in question_lower for word in ['compare', 'comparison', 'versus', 'vs', 'difference', 'between']):
        intent['type'] = 'comparison'
        intent['comparison'] = True
        
    elif any(word in question_lower for word in ['trend', 'growth', 'increase', 'decrease', 'change', 'over time']):
        intent['type'] = 'temporal'
        intent['temporal_analysis'] = True
        
    elif countries or continents or years:
        intent['type'] = 'specific_query'
        intent['specific_data'] = True
        
    elif any(word in question_lower for word in ['total', 'sum', 'global', 'world', 'overall']):
        intent['type'] = 'aggregate'
        
    else:
        intent['type'] = 'general'
    
    return intent

def extract_continents_from_query(query_lower, dataframe):
    """
    Extract continent names from the query.
    Also handles African regional queries by returning 'Africa' as the continent.
    """
    continents_in_data = dataframe['Continent'].unique()
    found_continents = []
    
    # Check for African regional queries first
    african_regions = ["west africa", "east africa", "north africa", "southern africa", "central africa"]
    for region in african_regions:
        if region in query_lower:
            if 'Africa' in continents_in_data:
                found_continents.append('Africa')
                return found_continents  # Return early for regional queries
    
    continent_aliases = {
        'africa': 'Africa', 'african': 'Africa',
        'asia': 'Asia', 'asian': 'Asia',
        'europe': 'Europe', 'european': 'Europe',
        'america': 'America', 'americas': 'America',
        'north america': 'North America', 'south america': 'South America'
    }
    
    for alias, continent in continent_aliases.items():
        if alias in query_lower and continent in continents_in_data:
            found_continents.append(continent)
    
    return found_continents

def execute_intelligent_analysis(intent, dataframe):
    """
    Steps 3, 4, 5: Execute analysis using dataset info, provide clear explanations, handle missing data
    """
    df = dataframe.copy()
    
    try:
        if intent['type'] == 'ranking':
            return handle_ranking_questions(intent, df)
            
        elif intent['type'] == 'comparison':
            return handle_comparison_questions(intent, df)
            
        elif intent['type'] == 'temporal':
            return handle_temporal_questions(intent, df)
            
        elif intent['type'] == 'specific_query':
            return handle_specific_queries(intent, df)
            
        elif intent['type'] == 'aggregate':
            return handle_aggregate_questions(intent, df)
            
        else:
            return handle_general_questions(intent, df)
            
    except Exception as e:
        return f"I couldn't complete the analysis due to: {str(e)}. Please try rephrasing your question."

def handle_ranking_questions(intent, df):
    """Handle 'which country has highest...' type questions"""
    action_text = intent['action'].lower()
    year_filter = intent['years'][0] if intent['years'] else None
    
    # Filter data based on action
    if intent['action'] == 'Export':
        filtered_df = df[df['Action'] == 'Export']
    elif intent['action'] == 'Import':
        filtered_df = df[df['Action'] == 'Import']
    else:
        filtered_df = df
    
    # Apply year filter if specified
    if year_filter:
        filtered_df = filtered_df[filtered_df['Year'] == year_filter]
        year_text = f" in {year_filter}"
        
        if filtered_df.empty:
            return f"I don't have any {action_text} data for {year_filter} in the dataset. The available years are {df['Year'].min()}-{df['Year'].max()}."
    else:
        year_text = " (1995-2021)"
    
    if filtered_df.empty:
        return f"I don't have any {action_text} data matching your criteria."
    
    # Calculate rankings
    country_totals = filtered_df.groupby('Country')['TradeValue'].sum().sort_values(ascending=False)
    
    if country_totals.empty:
        return f"No {action_text} data is available for the specified criteria."
    
    # Extract top N (default 5 unless specified)
    import re
    top_n_match = re.search(r'top\s+(\d+)', intent['original_question'])
    n = int(top_n_match.group(1)) if top_n_match else 5
    
    top_countries = country_totals.head(n)
    top_country = top_countries.index[0]
    top_value = top_countries.iloc[0]
    
    # Build precise, rule-compliant response
    year_specific = f" in {year_filter}" if year_filter else " (1995-2021)"
    
    if len(top_countries) == 1 or 'which country' in intent['original_question'] or 'highest' in intent['original_question']:
        # Single result response
        if intent['action'] == 'Export':
            explanation = f"**{top_country}** had the highest exports{year_specific}: {format_human_readable(top_value)}."
        elif intent['action'] == 'Import':
            explanation = f"**{top_country}** had the highest imports{year_specific}: {format_human_readable(top_value)}."
        else:
            explanation = f"**{top_country}** had the highest total trade{year_specific}: {format_human_readable(top_value)}."
    else:
        # Multiple results response
        explanation = f"**Top {n} {action_text.lower()}{year_specific} (by TradeValue, descending):** "
        for i, (country, value) in enumerate(top_countries.items(), 1):
            explanation += f"{i}. {country} ({format_human_readable(value)})"
            if i < len(top_countries):
                explanation += ", "
        explanation += "."
    
    return explanation

def handle_specific_queries(intent, df):
    """Handle specific country/continent/year queries"""
    result_parts = []
    
    # Process each country or continent
    entities = intent['countries'] + intent['continents']
    
    if not entities and not intent['years']:
        return "I need more specific information. Please mention a country, continent, or year to analyze."
    
    for entity in entities:
        # Check if it's a country or continent
        if entity in df['Country'].values:
            entity_data = df[df['Country'] == entity]
            entity_type = "country"
        elif entity in df['Continent'].values:
            entity_data = df[df['Continent'] == entity]
            entity_type = "continent"
        else:
            continue
            
        # Apply year filters
        if intent['years']:
            if len(intent['years']) == 1:
                year = intent['years'][0]
                entity_data = entity_data[entity_data['Year'] == year]
                time_text = f" in {year}"
            elif len(intent['years']) >= 2:
                start_year, end_year = min(intent['years']), max(intent['years'])
                entity_data = entity_data[
                    (entity_data['Year'] >= start_year) & 
                    (entity_data['Year'] <= end_year)
                ]
                time_text = f" from {start_year} to {end_year}"
            else:
                time_text = ""
        else:
            time_text = " (1995-2021)"
        
        if entity_data.empty:
            result_parts.append(f"No data is available for {entity}{time_text}.")
            continue
        
        # Apply operation-specific calculation based on enhanced rules
        if intent['operation'] == 'sum' and len(intent['years']) >= 2:
            # Sum across all years in range
            if intent['action'] == 'Export':
                total_value = entity_data[entity_data['Action'] == 'Export']['TradeValue'].sum()
                action_text = "exports"
            elif intent['action'] == 'Import':
                total_value = entity_data[entity_data['Action'] == 'Import']['TradeValue'].sum()
                action_text = "imports"
            else:
                total_value = entity_data['TradeValue'].sum()
                action_text = "total trade"
                
        elif intent['operation'] == 'compare' and len(intent['years']) >= 2:
            # Compare first vs last year only
            years_sorted = sorted(intent['years'])
            first_year, last_year = years_sorted[0], years_sorted[-1]
            
            first_data = entity_data[entity_data['Year'] == first_year]
            last_data = entity_data[entity_data['Year'] == last_year]
            
            if first_data.empty or last_data.empty:
                result_parts.append(f"Insufficient data to compare **{entity}** between {first_year} and {last_year}.")
                continue
                
            if intent['action'] == 'Export':
                first_val = first_data[first_data['Action'] == 'Export']['TradeValue'].sum()
                last_val = last_data[last_data['Action'] == 'Export']['TradeValue'].sum()
                action_text = "exports"
            elif intent['action'] == 'Import':
                first_val = first_data[first_data['Action'] == 'Import']['TradeValue'].sum()
                last_val = last_data[last_data['Action'] == 'Import']['TradeValue'].sum()
                action_text = "imports"
            else:
                first_val = first_data['TradeValue'].sum()
                last_val = last_data['TradeValue'].sum()
                action_text = "trade"
                
            if first_val == 0:
                result_parts.append(f"No {action_text} data for **{entity}** in {first_year}.")
                continue
                
            change = last_val - first_val
            change_pct = (change / first_val) * 100
            direction = "increased" if change > 0 else "decreased"
            
            explanation = f"**{entity}** {action_text} {direction} from {first_year} to {last_year}: {format_human_readable(abs(change))} ({abs(change_pct):.1f}%)."
            result_parts.append(explanation)
            continue
            
        else:
            # Standard lookup/calculation
            if intent['action'] == 'Export':
                total_value = entity_data[entity_data['Action'] == 'Export']['TradeValue'].sum()
                action_text = "exports"
            elif intent['action'] == 'Import':
                total_value = entity_data[entity_data['Action'] == 'Import']['TradeValue'].sum()
                action_text = "imports"
            else:
                total_value = entity_data['TradeValue'].sum()
                action_text = "total trade"
        
        # Build rule-compliant response
        explanation = f"**{entity}**{time_text}: {format_human_readable(total_value)} {action_text}."
        
        result_parts.append(explanation)
    
    return " ".join(result_parts) if result_parts else "No data found for the specified criteria."

def handle_aggregate_questions(intent, df):
    """Handle global/total questions"""
    year_filter_text = ""
    
    # Apply year filtering
    if intent['years']:
        if len(intent['years']) == 1:
            year = intent['years'][0]
            df = df[df['Year'] == year]
            year_filter_text = f" in {year}"
        elif len(intent['years']) >= 2:
            start_year, end_year = min(intent['years']), max(intent['years'])
            df = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)]
            year_filter_text = f" from {start_year} to {end_year}"
    else:
        year_filter_text = " (1995-2021)"
    
    if df.empty:
        return f"No data is available for the specified time period."
    
    # Calculate global totals
    total_exports = df[df['Action'] == 'Export']['TradeValue'].sum()
    total_imports = df[df['Action'] == 'Import']['TradeValue'].sum()
    total_trade = total_exports + total_imports
    
    explanation = f"**Global Crude Oil Trade{year_filter_text}:**\n"
    explanation += f"• Total Trade Value: ${total_trade/1e12:.3f} trillion USD\n"
    explanation += f"• Total Exports: ${total_exports/1e12:.3f} trillion USD\n"
    explanation += f"• Total Imports: ${total_imports/1e12:.3f} trillion USD\n"
    explanation += f"• Number of Countries: {df['Country'].nunique()}\n"
    explanation += f"• Number of Records: {len(df):,}"
    
    return explanation

def handle_general_questions(intent, df):
    """Handle general questions about the dataset"""
    total_records = len(df)
    countries = df['Country'].nunique()
    continents = df['Continent'].nunique() 
    years = f"{df['Year'].min()}-{df['Year'].max()}"
    total_value = df['TradeValue'].sum()
    
    explanation = f"**Crude Oil Trade Dataset Overview:**\n"
    explanation += f"• Time Period: {years}\n"
    explanation += f"• Total Records: {total_records:,}\n"
    explanation += f"• Countries Covered: {countries}\n"
    explanation += f"• Continents Covered: {continents}\n"
    explanation += f"• Total Trade Value: ${total_value/1e12:.3f} trillion USD\n\n"
    explanation += "You can ask me about:\n"
    explanation += "• Specific countries or years (e.g., 'Nigeria exports in 2004')\n"
    explanation += "• Rankings (e.g., 'Which country has the highest exports?')\n"
    explanation += "• Comparisons (e.g., 'Compare trade between 2000 and 2010')\n"
    explanation += "• Trade patterns and trends"
    
    return explanation

def analyze_dataframe_query(question, dataframe):
    """
    Legacy function - now redirects to the intelligent analyzer
    """
    return intelligent_dataframe_analyzer(question, dataframe)

def handle_comparison_questions(intent, df):
    """Handle comparison type questions"""
    if len(intent['years']) >= 2:
        year1, year2 = intent['years'][0], intent['years'][1]
        year1_data = df[df['Year'] == year1]['TradeValue'].sum()
        year2_data = df[df['Year'] == year2]['TradeValue'].sum()
        
        if year1_data == 0 or year2_data == 0:
            return f"Insufficient data for comparison between {year1} and {year2}."
        
        change = year2_data - year1_data
        change_pct = (change / year1_data) * 100
        
        explanation = f"**Crude Oil Trade Comparison:**\n"
        explanation += f"• {year1}: ${year1_data/1e12:.3f} trillion USD\n"
        explanation += f"• {year2}: ${year2_data/1e12:.3f} trillion USD\n"
        explanation += f"• Change: ${change/1e12:.3f} trillion USD ({change_pct:+.1f}%)\n"
        
        if change_pct > 0:
            explanation += f"\nTrade increased by {abs(change_pct):.1f}% from {year1} to {year2}."
        else:
            explanation += f"\nTrade decreased by {abs(change_pct):.1f}% from {year1} to {year2}."
        
        return explanation
    
    return "I need at least two years to make a comparison. Please specify the years you want to compare."

def handle_temporal_questions(intent, df):
    """
    Handle trend and temporal analysis questions with enhanced year range logic:
    - If question mentions "total" or "overall", calculate sum across all years in range
    - If question implies change/comparison ("increase", "difference"), compare first and last year only
    """
    if not intent['years'] or len(intent['years']) < 2:
        return "Please specify a year range for temporal analysis (e.g., 'between 2000 and 2010')."
    
    start_year, end_year = min(intent['years']), max(intent['years'])
    period_data = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)]
    
    if period_data.empty:
        return f"No data available for the period {start_year}-{end_year}."
    
    # Determine if user wants total sum or year-to-year comparison
    question_lower = str(intent.get('original_question', '')).lower()
    is_total_request = any(word in question_lower for word in ['total', 'overall', 'sum', 'combined'])
    is_comparison_request = any(word in question_lower for word in ['increase', 'decrease', 'change', 'difference', 'growth'])
    
    if is_total_request and not is_comparison_request:
        # Calculate total sum across all years in range
        total_value = period_data['TradeValue'].sum()
        return f"**Total crude oil trade ({start_year}-{end_year}):** {format_human_readable(total_value)}."
    
    else:
        # Compare first and last year only
        yearly_totals = period_data.groupby('Year')['TradeValue'].sum().sort_index()
        
        first_value = yearly_totals.loc[start_year] if start_year in yearly_totals.index else 0
        last_value = yearly_totals.loc[end_year] if end_year in yearly_totals.index else 0
        
        if first_value == 0 or last_value == 0:
            return f"Insufficient data to compare {start_year} and {end_year}."
        
        change = last_value - first_value
        change_pct = (change / first_value) * 100
        
        direction = "increased" if change > 0 else "decreased"
        return f"**Trade {direction} from {start_year} to {end_year}:** {format_human_readable(abs(change))} ({abs(change_pct):.1f}%)."

def extract_countries_from_query(query_lower, dataframe):
    """
    Extract country names from the query by matching against actual countries in the dataset.
    Prioritizes exact matches when there are similar country names (e.g., Niger vs Nigeria).
    Also handles regional queries like "East Africa", "West Africa", etc.
    """
    countries_in_data = dataframe['Country'].unique()
    found_countries = []
    exact_matches = []
    partial_matches = []
    
    # Check for regional queries first
    for region, countries in region_map.items():
        if region in query_lower:
            # Find countries from this region that exist in our dataset
            regional_countries = [country for country in countries if country in countries_in_data]
            if regional_countries:
                return regional_countries
    
    # If no regional match, proceed with individual country matching
    for country in countries_in_data:
        country_lower = country.lower()
        
        # Check for exact word match first (e.g., "niger" matches exactly, not as part of "nigeria")
        import re
        if re.search(rf'\b{re.escape(country_lower)}\b', query_lower):
            exact_matches.append(country)
        # Check for partial match (country name appears in query)
        elif country_lower in query_lower:
            partial_matches.append(country)
    
    # Prioritize exact matches over partial matches
    if exact_matches:
        found_countries = exact_matches
    elif partial_matches:
        found_countries = partial_matches
    
    return found_countries

def create_data_summary(dataframe):
    """
    Create a comprehensive but concise summary of the dataframe for AI analysis
    """
    try:
        # Basic stats
        total_records = len(dataframe)
        date_range = f"{dataframe['Year'].min()}-{dataframe['Year'].max()}"
        countries = dataframe['Country'].nunique()
        total_trade_value = dataframe['TradeValue'].sum()
        
        # Top exporters and importers
        top_exporters = dataframe[dataframe['Action'] == 'Export'].groupby('Country')['TradeValue'].sum().sort_values(ascending=False).head(5)
        top_importers = dataframe[dataframe['Action'] == 'Import'].groupby('Country')['TradeValue'].sum().sort_values(ascending=False).head(5)
        
        # Yearly totals (sample years for context)
        yearly_sample = dataframe.groupby('Year')['TradeValue'].sum().head(10)  # First 10 years
        
        # Country-specific examples
        nigeria_exports = dataframe[(dataframe['Country'] == 'Nigeria') & (dataframe['Action'] == 'Export')]
        nigeria_by_year = nigeria_exports.groupby('Year')['TradeValue'].sum().head(5)
        
        summary = f"""
CRUDE OIL TRADE DATA SUMMARY ({date_range})
==========================================
Total Records: {total_records:,}
Countries: {countries}
Total Trade Value: {format_human_readable(total_trade_value)}

TOP 5 EXPORTERS (Total 1995-2021):
"""
        for i, (country, value) in enumerate(top_exporters.items(), 1):
            summary += f"{i}. {country}: {format_human_readable(value)}\n"
        
        summary += "\nTOP 5 IMPORTERS (Total 1995-2021):\n"
        for i, (country, value) in enumerate(top_importers.items(), 1):
            summary += f"{i}. {country}: {format_human_readable(value)}\n"
        
        summary += f"\nSAMPLE YEARLY TOTALS:\n"
        for year, value in yearly_sample.items():
            summary += f"{year}: {format_human_readable(value)}\n"
        
        if not nigeria_by_year.empty:
            summary += f"\nNIGERIA EXPORTS (Sample Years):\n"
            for year, value in nigeria_by_year.items():
                summary += f"{year}: $" + f"{value/1e9:.2f}B USD\n"
        
        return summary
        
    except Exception as e:
        return f"Data summary error: {str(e)}"
        
        # Create a comprehensive pandas analysis tool with full dataframe access
        def analyze_full_dataframe(query: str) -> str:
            """
            Full pandas dataframe analyzer with complete access to all operations
            """
            try:
                import re
                import numpy as np
                
                # Make dataframe available for analysis
                df = dataframe.copy()
                
                # Enhanced query processing with full pandas capabilities
                query_lower = query.lower()
                
                # Country-specific analysis
                if any(country in query_lower for country in ['nigeria', 'angola', 'algeria', 'egypt']):
                    # Extract country name
                    countries = ['nigeria', 'angola', 'algeria', 'egypt', 'gabon', 'cameroon']
                    found_country = None
                    for country in countries:
                        if country in query_lower:
                            found_country = country.title()
                            break
                    
                    if found_country:
                        country_data = df[df['Country'] == found_country]
                        
                        # Year-specific analysis
                        years = [int(year) for year in re.findall(r'\b(?:19|20)\d{2}\b', query)]
                        if years:
                            year = years[0]
                            year_data = country_data[country_data['Year'] == year]
                            
                            if 'export' in query_lower:
                                exports = year_data[year_data['Action'] == 'Export']
                                if not exports.empty:
                                    value = exports['TradeValue'].sum()
                                    return f"**{found_country}** in {year}: {format_human_readable(value)} exports."
                            elif 'import' in query_lower:
                                imports = year_data[year_data['Action'] == 'Import']
                                if not imports.empty:
                                    value = imports['TradeValue'].sum()
                                    return f"**{found_country}** in {year}: {format_human_readable(value)} imports."
                            else:
                                total_value = year_data['TradeValue'].sum()
                                return f"**{found_country}** in {year}: {format_human_readable(total_value)} total trade."
                        
                        # Multi-year analysis for country with enhanced logic
                        elif len(years) >= 2 or 'between' in query_lower:
                            if len(years) >= 2:
                                start_year, end_year = min(years), max(years)
                            else:
                                # Default range if no specific years mentioned
                                start_year, end_year = 2000, 2010
                                
                            period_data = country_data[
                                (country_data['Year'] >= start_year) & 
                                (country_data['Year'] <= end_year)
                            ]
                            
                            if period_data.empty:
                                return f"No data for **{found_country}** between {start_year} and {end_year}."
                            
                            # Enhanced year range logic
                            is_total_request = any(word in query_lower for word in ['total', 'overall', 'sum', 'combined'])
                            is_comparison_request = any(word in query_lower for word in ['increase', 'decrease', 'change', 'difference', 'growth'])
                            
                            if is_total_request and not is_comparison_request:
                                # Calculate total sum across all years in range
                                if 'export' in query_lower:
                                    exports = period_data[period_data['Action'] == 'Export']
                                    total_value = exports['TradeValue'].sum()
                                    return f"**{found_country}** total exports ({start_year}-{end_year}): {format_human_readable(total_value)}."
                                elif 'import' in query_lower:
                                    imports = period_data[period_data['Action'] == 'Import']
                                    total_value = imports['TradeValue'].sum()
                                    return f"**{found_country}** total imports ({start_year}-{end_year}): {format_human_readable(total_value)}."
                                else:
                                    total_value = period_data['TradeValue'].sum()
                                    return f"**{found_country}** total trade ({start_year}-{end_year}): {format_human_readable(total_value)}."
                            
                            else:
                                # Compare first and last year only
                                start_data = country_data[country_data['Year'] == start_year]
                                end_data = country_data[country_data['Year'] == end_year]
                                
                                if start_data.empty or end_data.empty:
                                    return f"Insufficient data to compare **{found_country}** between {start_year} and {end_year}."
                                
                                if 'export' in query_lower:
                                    start_val = start_data[start_data['Action'] == 'Export']['TradeValue'].sum()
                                    end_val = end_data[end_data['Action'] == 'Export']['TradeValue'].sum()
                                    action_text = "exports"
                                elif 'import' in query_lower:
                                    start_val = start_data[start_data['Action'] == 'Import']['TradeValue'].sum()
                                    end_val = end_data[end_data['Action'] == 'Import']['TradeValue'].sum()
                                    action_text = "imports"
                                else:
                                    start_val = start_data['TradeValue'].sum()
                                    end_val = end_data['TradeValue'].sum()
                                    action_text = "trade"
                                
                                if start_val == 0:
                                    return f"No {action_text} data for **{found_country}** in {start_year}."
                                
                                change = end_val - start_val
                                change_pct = (change / start_val) * 100
                                direction = "increased" if change > 0 else "decreased"
                                
                                return f"**{found_country}** {action_text} {direction} from {start_year} to {end_year}: {format_human_readable(abs(change))} ({abs(change_pct):.1f}%)."
                        
                        # Overall country statistics
                        else:
                            exports = country_data[country_data['Action'] == 'Export']['TradeValue'].sum()
                            imports = country_data[country_data['Action'] == 'Import']['TradeValue'].sum()
                            years_active = sorted(country_data['Year'].unique())
                            return f"{found_country} (1995-2021): Exports $" + f"{exports/1e9:.2f}B, Imports $" + f"{imports/1e9:.2f}B, Active years: {len(years_active)}"
                
                # Global trade analysis
                elif 'total trade' in query_lower or 'global' in query_lower:
                    years = [int(year) for year in re.findall(r'\b(?:19|20)\d{2}\b', query)]
                    if len(years) >= 2:
                        start_year, end_year = min(years), max(years)
                        period_data = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)]
                        total_value = period_data['TradeValue'].sum()
                        return f"Global crude oil trade ({start_year}-{end_year}): $" + f"{total_value/1e12:.3f}T USD"
                    elif len(years) == 1:
                        year = years[0]
                        year_data = df[df['Year'] == year]
                        total_value = year_data['TradeValue'].sum()
                        return f"Global crude oil trade in {year}: $" + f"{total_value/1e12:.3f}T USD"
                
                # Top countries analysis
                elif 'top' in query_lower or 'largest' in query_lower or 'biggest' in query_lower:
                    if 'export' in query_lower:
                        exports_by_country = df[df['Action'] == 'Export'].groupby('Country')['TradeValue'].sum().sort_values(ascending=False)
                        top_5 = exports_by_country.head(5)
                        result = "Top 5 crude oil exporters:\n"
                        for i, (country, value) in enumerate(top_5.items(), 1):
                            result += f"{i}. {country}: $" + f"{value/1e9:.1f}B USD\n"
                        return result.strip()
                    elif 'import' in query_lower:
                        imports_by_country = df[df['Action'] == 'Import'].groupby('Country')['TradeValue'].sum().sort_values(ascending=False)
                        top_5 = imports_by_country.head(5)
                        result = "Top 5 crude oil importers:\n"
                        for i, (country, value) in enumerate(top_5.items(), 1):
                            result += f"{i}. {country}: $" + f"{value/1e9:.1f}B USD\n"
                        return result.strip()
                
                # Year comparison
                elif 'compare' in query_lower:
                    years = [int(year) for year in re.findall(r'\b(?:19|20)\d{2}\b', query)]
                    if len(years) >= 2:
                        year1, year2 = years[0], years[1]
                        year1_total = df[df['Year'] == year1]['TradeValue'].sum()
                        year2_total = df[df['Year'] == year2]['TradeValue'].sum()
                        change_pct = ((year2_total - year1_total) / year1_total) * 100
                        return f"Trade comparison: {year1}: $" + f"{year1_total/1e12:.3f}T, {year2}: $" + f"{year2_total/1e12:.3f}T (Change: {change_pct:+.1f}%)"
                
                # Trend analysis
                elif 'trend' in query_lower or 'growth' in query_lower:
                    yearly_totals = df.groupby('Year')['TradeValue'].sum()
                    first_year_value = yearly_totals.iloc[0]
                    last_year_value = yearly_totals.iloc[-1]
                    total_growth = ((last_year_value - first_year_value) / first_year_value) * 100
                    return f"Crude oil trade trend (1995-2021): Started at $" + f"{first_year_value/1e12:.3f}T, ended at $" + f"{last_year_value/1e12:.3f}T (Total growth: {total_growth:+.1f}%)"
                
                # Default comprehensive summary
                else:
                    total_records = len(df)
                    total_value = df['TradeValue'].sum()
                    unique_countries = df['Country'].nunique()
                    year_range = f"{df['Year'].min()}-{df['Year'].max()}"
                    return f"Crude Oil Trade Database: {total_records:,} records, {unique_countries} countries, {year_range}, Total value: $" + f"{total_value/1e12:.3f}T USD"
                
            except Exception as e:
                return f"Analysis error: {str(e)}"
        
        # Create the pandas analysis tool
        pandas_tool = Tool(
            name="CrudeOilDataAnalyzer", 
            description="Comprehensive crude oil trade data analyzer with full pandas dataframe access. Can analyze any aspect of the data including country-specific trades, yearly comparisons, trends, and global statistics.",
            func=analyze_full_dataframe
        )
        
        # Use the tool directly for now (simpler than full agent setup)
        result = analyze_full_dataframe(question)
        return result
        
    except Exception as e:
        return f"LangChain analysis failed: {str(e)}"

def get_basic_response(question, dataframe):
    """
    Simple direct answer system - no more garbled text!
    """
    import re
    question_lower = question.lower()
    
    # Extract years
    years = [int(year) for year in re.findall(r'\b(?:19|20)\d{2}\b', question)]
    
    # Handle specific cases with direct answers
    if 'nigeria' in question_lower and 'export' in question_lower and years:
        year = years[0]
        nigeria_exports = dataframe[(dataframe['Country'] == 'Nigeria') & 
                                  (dataframe['Action'] == 'Export') & 
                                  (dataframe['Year'] == year)]
        if not nigeria_exports.empty:
            value = nigeria_exports['TradeValue'].sum()
            return f"Nigeria's exports in {year}: ${value/1e9:.2f}B USD"
    
    return "Please try rephrasing your question or check that the data exists for the requested year/country."

def get_openai_response(question, dataframe):
    """
    Generate AI response using OpenAI API with comprehensive data analysis
    """
    import re
    question_lower = question.lower()
    
    # Perform actual data analysis based on the question
    analysis_results = perform_data_analysis(question, dataframe)
    
    # Create comprehensive context with real data
    context = f"""You are an expert crude oil trade analyst. Use ONLY the specific data analysis results provided below to answer the user's question. DO NOT provide generic responses.

QUESTION: {question}

DATA ANALYSIS RESULTS:
{analysis_results}

INSTRUCTIONS:
1. Answer the specific question using ONLY the data provided above
2. Use exact numbers and statistics from the analysis results
3. Be precise and data-driven
4. Format large numbers with appropriate suffixes (T for trillion, B for billion, M for million)
5. Focus on insights relevant to the specific question asked"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": f"Based on the data analysis results provided, answer this question: {question}"}
            ],
            max_tokens=600,
            temperature=0.1
        )
        
        if response and response.choices:
            return response.choices[0].message.content.strip()
        else:
            raise Exception("Empty response from OpenAI")
            
    except Exception as e:
        raise Exception(f"OpenAI API error: {str(e)}")

def perform_data_analysis(question, df):
    """
    Perform comprehensive data analysis based on the question content
    """
    import re
    question_lower = question.lower()
    results = []
    
    # Extract years from question
    year_pattern = r'\b(?:19|20)\d{2}\b'
    years_in_question = [int(year) for year in re.findall(year_pattern, question)]
    
    # Extract year ranges (e.g., "between 2000 and 2002")
    range_pattern = r'between\s+(\d{4})\s+and\s+(\d{4})'
    range_match = re.search(range_pattern, question_lower)
    if range_match:
        start_year, end_year = int(range_match.group(1)), int(range_match.group(2))
        years_in_question = list(range(start_year, end_year + 1))
    
    # Extract action from question (export/import)
    action_mentioned = None
    if 'export' in question_lower:
        action_mentioned = 'Export'
    elif 'import' in question_lower:
        action_mentioned = 'Import'
    
    # Handle specific action + year questions (e.g., "highest export in 2004")
    if years_in_question and action_mentioned:
        filtered_df = df[(df['Year'].isin(years_in_question)) & (df['Action'] == action_mentioned)]
        if not filtered_df.empty:
            total_value = filtered_df['TradeValue'].sum()
            results.append(f"Total {action_mentioned.lower()} value for {years_in_question}: ${total_value/1e12:.3f}T USD")
            
            # Top countries for this specific action and year(s)
            top_countries = filtered_df.groupby('Country')['TradeValue'].sum().nlargest(10)
            results.append(f"Top 10 countries by {action_mentioned.lower()} value:")
            for i, (country, value) in enumerate(top_countries.items(), 1):
                results.append(f"  {i}. {country}: ${value/1e9:.2f}B USD")
                
            # Specific answer for "which country" questions
            if any(word in question_lower for word in ['which country', 'what country', 'who', 'highest']):
                top_country = top_countries.index[0]
                top_value = top_countries.iloc[0]
                results.insert(0, f"Answer: {top_country} had the highest {action_mentioned.lower()} in {years_in_question[0] if len(years_in_question) == 1 else f'{years_in_question[0]}-{years_in_question[-1]}'} with ${top_value/1e9:.2f}B USD")
    
    # If years are mentioned but no specific action - give DIRECT answer
    elif years_in_question:
        filtered_df = df[df['Year'].isin(years_in_question)]
        if not filtered_df.empty:
            total_value = filtered_df['TradeValue'].sum()
            
            # For simple "trade value between X and Y" questions, give direct answer
            if 'trade value' in question_lower and len(years_in_question) > 1:
                results.append(f"The total trade value between {years_in_question[0]} and {years_in_question[-1]} was ${total_value/1e12:.3f}T USD")
            
            # For single year questions  
            elif len(years_in_question) == 1:
                results.append(f"Total trade value for {years_in_question[0]}: ${total_value/1e12:.3f}T USD")
                
            # Only add details if specifically asked for breakdown/analysis
            if any(word in question_lower for word in ['breakdown', 'analysis', 'details', 'countries', 'top']):
                results.append(f"Number of trade records: {len(filtered_df):,}")
                
                # Top countries by trade value
                top_countries = filtered_df.groupby('Country')['TradeValue'].sum().nlargest(5)
                results.append("Top 5 countries by trade value:")
                for i, (country, value) in enumerate(top_countries.items(), 1):
                    results.append(f"  {i}. {country}: ${value/1e9:.2f}B USD")
    
    # Check for country-specific questions
    countries_mentioned = []
    unique_countries = df['Country'].unique()
    for country in unique_countries:
        if country.lower() in question_lower:
            countries_mentioned.append(country)
    
    if countries_mentioned:
        for country in countries_mentioned:
            country_data = df[df['Country'] == country]
            total_trade = country_data['TradeValue'].sum()
            results.append(f"{country} total trade value: ${total_trade/1e12:.3f}T USD")
            
            # Year range for this country
            years = sorted(country_data['Year'].unique())
            results.append(f"{country} data available: {years[0]}-{years[-1]} ({len(country_data):,} records)")
    
    # Check for continent analysis
    continents = ['africa', 'asia', 'europe', 'north america', 'south america', 'oceania']
    for continent in continents:
        if continent in question_lower:
            continent_data = df[df['Continent'].str.contains(continent, case=False, na=False)]
            if not continent_data.empty:
                total_trade = continent_data['TradeValue'].sum()
                results.append(f"{continent.title()} total trade: ${total_trade/1e12:.3f}T USD")
    
    # Check for import/export specific questions
    if 'import' in question_lower or 'export' in question_lower:
        by_action = df.groupby('Action')['TradeValue'].sum()
        results.append("Trade breakdown by action:")
        for action, value in by_action.items():
            results.append(f"  {action}: ${value/1e12:.3f}T USD")
    
    # If no specific analysis was triggered, provide general insights
    if not results:
        total_trade = df['TradeValue'].sum()
        results.append(f"Total dataset trade value: ${total_trade/1e12:.2f}T USD")
        results.append(f"Data covers {df['Year'].min()}-{df['Year'].max()} ({len(df):,} records)")
        
        # Top countries overall
        top_countries = df.groupby('Country')['TradeValue'].sum().nlargest(5)
        results.append("Top 5 countries by total trade value:")
        for i, (country, value) in enumerate(top_countries.items(), 1):
            results.append(f"  {i}. {country}: ${value/1e12:.3f}T USD")
    
    return "\n".join(results)

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


