import streamlit as st
import pandas as pd
import joblib
import json
import time
import requests
import numpy as np

# --- PAGE CONFIGURATION (Must be first) ---
st.set_page_config(
    page_title="Social Impact AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CONFIGURATION ---
MODEL_FILE = 'mental_health_model (6).joblib' 
GEMINI_MODEL = 'gemini-2.5-flash'
API_KEY = st.secrets.get("GEMINI_API_KEY", None)

# --- CUSTOM CSS (Modern Glassmorphism Theme) ---
st.markdown("""
<style>
    /* IMPORT FONT */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    /* GLOBAL THEME */
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0f172a 0%, #1e1b4b 90%);
        color: #e2e8f0;
        font-family: 'Outfit', sans-serif;
    }

    /* REMOVE DEFAULT PADDING */
    div.block-container {
        padding-top: 2rem;
        padding-bottom: 5rem;
    }

    /* SIDEBAR STYLING */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.6);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
    }
    
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: #94a3b8 !important;
        font-weight: 600;
        letter-spacing: 0.05em;
    }

    /* INPUT FIELDS (Glass Style) */
    .stTextInput > div > div > input, 
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div {
        background-color: rgba(255, 255, 255, 0.05) !important;
        color: #f8fafc !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
    }
    
    /* INPUT FOCUS STATE */
    .stTextInput > div > div > input:focus, 
    .stNumberInput > div > div > input:focus {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 10px rgba(56, 189, 248, 0.2) !important;
    }

    /* BUTTONS */
    .stButton > button {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.2rem !important;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.5);
    }

    /* CUSTOM CARDS */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 2rem;
        backdrop-filter: blur(10px);
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }

    /* HEADERS */
    .gradient-text {
        background: linear-gradient(to right, #22d3ee, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    
    /* SCORE BOX */
    .score-container {
        text-align: center;
        padding: 2rem;
        border-radius: 20px;
        background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, rgba(0,0,0,0) 70%);
        border: 1px solid rgba(255,255,255,0.1);
        margin-top: 2rem;
    }
    
    /* LIST ITEMS IN RESULTS */
    ul {
        list-style-type: none; /* Remove bullets */
        padding: 0;
        margin: 0;
    }
    
    li {
        padding-left: 1.5em; 
        text-indent: -1.5em;
        margin-bottom: 8px;
        color: #cbd5e1;
    }
    
    li::before {
        content: "▹";
        color: #38bdf8;
        padding-right: 8px;
        font-weight: bold;
    }

</style>
""", unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_resource
def load_ml_model():
    try:
        model = joblib.load(MODEL_FILE)
        return model
    except Exception as e:
        # We don't want the app to crash visually if the model file is missing for the demo
        return None

model = load_ml_model()

# --- GEMINI API CALL HANDLER ---
def call_gemini(prompt, is_json=True, max_retries=3):
    if not API_KEY:
        st.warning("⚠️ Gemini API key missing. AI features disabled.")
        return None
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={API_KEY}"
    
    for i in range(max_retries):
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        if is_json:
            payload["generationConfig"] = {"responseMimeType": "application/json"}
        try:
            response = requests.post(url, headers={'Content-Type': 'application/json'}, json=payload, timeout=20)
            if response.status_code == 200:
                data = response.json()
                return data.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text')
            time.sleep(1)
        except Exception:
            pass
    return None

# --- ML FEATURE PREP ---
MODEL_COLUMNS = [
    'Age', 'Gender', 'Academic_Level', 'Avg_Daily_Usage_Hours', 'Affects_Academic_Performance', 'Sleep_Hours_Per_Night', 'Conflicts_Over_Social_Media', 'Addicted_Score', 'Most_Used_Platform_Facebook', 'Most_Used_Platform_Instagram', 'Most_Used_Platform_KakaoTalk', 'Most_Used_Platform_LINE', 'Most_Used_Platform_LinkedIn', 'Most_Used_Platform_Snapchat', 'Most_Used_Platform_TikTok', 'Most_Used_Platform_Twitter', 'Most_Used_Platform_VKontakte', 'Most_Used_Platform_WeChat', 'Most_Used_Platform_WhatsApp', 'Most_Used_Platform_YouTube', 'Most_Used_Platform_Youtube', 'Relationship_Status_Complicated', 'Relationship_Status_In Relationship', 'Relationship_Status_Single']

# --- SIDEBAR UI ---
with st.sidebar:
    st.markdown("### 👤 User Profile")
    age = st.number_input("Age", 10, 100, 20)
    gender = st.selectbox("Gender", ["Male", "Female"])
    academic_level = st.selectbox("Academic Level", ["High School", "Undergraduate", "Graduate", "Middle School"])
    
    st.markdown("---")
    st.markdown("### 📱 Digital Habits")
    avg_daily_usage = st.number_input("Daily Hours", 0.0, 24.0, 4.0, 0.5)
    platform = st.selectbox("Main Platform", ["TikTok", "YouTube", "Instagram", "Twitter", "Facebook", "Snapchat", "WhatsApp", "LinkedIn"])
    addiction = st.slider("Self-Perceived Addiction (1-10)", 1, 10, 5)
    
    st.markdown("---")
    st.markdown("### ❤️ Well-being")
    sleep = st.number_input("Sleep Hours", 0.0, 24.0, 7.0, 0.5)
    affects_perf = st.radio("Impacts Academics?", ["No", "Yes"], horizontal=True)
    conflicts = st.number_input("Conflicts caused by Social Media", 0, 10, 0)
    rel_status = st.selectbox("Relationship Status", ["Single", "In a relationship", "Married", "Divorced"])
    
    st.markdown("<br>", unsafe_allow_html=True)
    calculate_button = st.button("🚀 Analyze Impact")

# --- MAIN UI ---
st.markdown('<h1 style="font-size: 3.5rem; margin-bottom: 0;">Social <span class="gradient-text">Impact</span></h1>', unsafe_allow_html=True)
st.markdown('<p style="font-size: 1.2rem; color: #94a3b8; margin-bottom: 2rem;">Analyze the invisible footprint of digital consumption on your mental wellness.</p>', unsafe_allow_html=True)

# --- APP LOGIC ---
if calculate_button:
    # --- DUMMY PREDICTION LOGIC (Fallback if model file missing) ---
    st.session_state.ai_results = {} 
    
    # Prepare Data
    input_df = pd.DataFrame(0, index=[0], columns=MODEL_COLUMNS)
    try:
        input_df['Gender'] = 1 if gender == "Female" else 0 
        input_df['Age'] = age
        input_df['Academic_Level'] = 1 # Simplified for demo
        input_df['Avg_Daily_Usage_Hours'] = avg_daily_usage
        input_df['Addicted_Score'] = addiction
        input_df['Conflicts_Over_Social_Media'] = conflicts
        input_df['Affects_Academic_Performance'] = 1 if affects_perf == "Yes" else 0
        
        # Safe Prediction
        if model:
            # Handle one-hot encoding columns safely
            plat_col = f"Most_Used_Platform_{platform}"
            if plat_col in MODEL_COLUMNS: input_df[plat_col] = 1
            
            wellness_score = model.predict(input_df)[0]
        else:
            # Mock calculation for UI demo purposes if file is missing
            base_score = 10 - (avg_daily_usage * 0.3) - (addiction * 0.2) + (sleep * 0.2)
            wellness_score = max(1, min(10, base_score)) # Clamp between 1 and 10
            if model is None: st.warning("⚠️ Model file not found. Using simulation logic for UI demo.")

        st.session_state['score'] = wellness_score
        st.session_state['user_data_ai'] = {"Age": age, "Hours": avg_daily_usage, "Platform": platform, "Addiction": addiction, "Sleep": sleep}
    except Exception as e:
        st.error(f"Prediction Error: {e}")

# --- RESULTS DISPLAY ---
if 'score' in st.session_state:
    score = st.session_state['score']
    
    # Determine Color based on score
    score_color = "#ef4444" if score < 4 else "#f59e0b" if score < 7 else "#10b981"
    
    # Score Section
    st.markdown(f"""
    <div class="score-container">
        <h3 style="margin:0; color:#94a3b8; text-transform:uppercase; letter-spacing:2px; font-size:1rem;">Calculated Wellness Score</h3>
        <h1 style="font-size: 6rem; font-weight: 800; margin: 0; color: {score_color}; text-shadow: 0 0 20px {score_color}40;">
            {score:.1f}<span style="font-size:2rem; color:#64748b;">/10</span>
        </h1>
        <p style="color: #cbd5e1;">Based on your digital habits and sleep patterns.</p>
    </div>
    """, unsafe_allow_html=True)

    # AI Section
    st.markdown('<br><h3 class="gradient-text">✨ AI Command Center</h3>', unsafe_allow_html=True)
    
    data = st.session_state['user_data_ai']
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div style="text-align:center;">', unsafe_allow_html=True)
        if st.button("📊 Deep Analysis"):
            with st.spinner("Analyzing patterns..."):
                prompt = f"Analyze this user data: {json.dumps(data)}. Return JSON with keys: 'persona' (2 word archetype), 'analysis' (2 sentences), 'tips' (list of 2 actionable tips)."
                res = call_gemini(prompt)
                if res: st.session_state.ai_results['analysis'] = json.loads(res); st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div style="text-align:center;">', unsafe_allow_html=True)
        if st.button("🕰️ Future Self"):
            with st.spinner("Connecting to 2029..."):
                prompt = f"Write a short, dramatic warning or congratulatory note from this user's future self in 2029 based on their current habits: {json.dumps(data)}. Max 60 words."
                res = call_gemini(prompt, is_json=False)
                if res: st.session_state.ai_results['future'] = res; st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown('<div style="text-align:center;">', unsafe_allow_html=True)
        if st.button("🍃 3-Day Detox"):
            with st.spinner("Generating plan..."):
                prompt = f"Create a 3-day social media detox plan for a {data['Platform']} user. Return JSON: {{'days': [{{'day': 'Day 1', 'theme': 'Theme', 'tasks': ['Task 1', 'Task 2']}}]}}"
                res = call_gemini(prompt)
                if res: st.session_state.ai_results['detox'] = json.loads(res); st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # Results Render Logic
    if st.session_state.get('ai_results'):
        st.markdown("<br>", unsafe_allow_html=True)
        
        results = st.session_state.ai_results
        
        # Analysis Card
        if 'analysis' in results:
            r = results['analysis']
            st.markdown(f"""
            <div class="glass-card">
                <div style="display:flex; align-items:center; gap:10px; margin-bottom:10px;">
                    <span style="font-size:1.5rem;">📊</span>
                    <h3 style="margin:0; color:#38bdf8;">Analysis: {r.get('persona', 'User')}</h3>
                </div>
                <p style="color:#e2e8f0; font-style:italic;">"{r.get('analysis', '')}"</p>
                <hr style="border-color:rgba(255,255,255,0.1);">
                <p style="font-weight:bold; color:#94a3b8; font-size:0.9rem;">RECOMMENDATIONS</p>
                <ul>
                    {"".join([f"<li>{t}</li>" for t in r.get('tips', [])])}
                </ul>
            </div>
            """, unsafe_allow_html=True)

        # Future Self Card
        if 'future' in results:
            st.markdown(f"""
            <div class="glass-card" style="border-left: 4px solid #f59e0b;">
                <div style="display:flex; align-items:center; gap:10px; margin-bottom:10px;">
                    <span style="font-size:1.5rem;">🕰️</span>
                    <h3 style="margin:0; color:#f59e0b;">Message from 2029</h3>
                </div>
                <p style="font-family: monospace; color:#fcd34d; background:rgba(0,0,0,0.2); padding:1rem; border-radius:8px;">
                    > INCOMING TRANSMISSION...<br><br>{results['future']}
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Detox Card
        if 'detox' in results:
            r = results['detox']
            days_html = ""
            for d in r.get('days', []):
                tasks_html = "".join([f"<li>{t}</li>" for t in d.get('tasks', [])])
                days_html += f"""
                <div style="background:rgba(255,255,255,0.05); padding:1rem; border-radius:8px; margin-bottom:10px;">
                    <strong style="color:#a78bfa;">{d.get('day')}: {d.get('theme')}</strong>
                    <ul style="margin-top:5px;">{tasks_html}</ul>
                </div>
                """
            
            st.markdown(f"""
            <div class="glass-card">
                <div style="display:flex; align-items:center; gap:10px; margin-bottom:15px;">
                    <span style="font-size:1.5rem;">🍃</span>
                    <h3 style="margin:0; color:#a78bfa;">Digital Detox Protocol</h3>
                </div>
                {days_html}
            </div>
            """, unsafe_allow_html=True)

st.markdown("<br><br><div style='text-align:center; color:#475569; font-size:0.8rem;'>AI Powered Social Impact Dashboard • Streamlit</div>", unsafe_allow_html=True)
