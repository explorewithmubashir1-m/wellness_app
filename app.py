import streamlit as st
import pandas as pd
import joblib
import json
import time
import requests
import numpy as np

# --- CONFIGURATION ---
MODEL_FILE = 'mental_health_model.joblib (6)' 
GEMINI_MODEL = 'gemini-2.5-flash-preview-09-2025'
# For deployment, fetch securely from Streamlit Cloud secrets.
API_KEY = st.secrets.get("GEMINI_API_KEY", None)
# --- CUSTOM CSS (High-Visibility Neo-Brutalist Theme) ---
st.markdown("""
<style>
    /* Global Background: Dark Purplish Gradient */
    .stApp {
        background: linear-gradient(135deg, #020617, #1e1b4b, #0f172a);
        color: #e2e8f0;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    /* Abstract background dots */
    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        background-image: radial-gradient(circle, rgba(255,255,255,0.1) 1px, transparent 1px);
        background-size: 24px 24px;
        pointer-events: none;
        z-index: 0;
    }

    /* Main Container Padding */
    div.block-container {
        padding-top: 2rem;
        max-width: 900px;
        position: relative;
        z-index: 1;
    }
    
    /* Neo-Brutalist Card: White background, Forced Black Text */
    div.stForm, .result-card, div[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border: 4px solid #000000 !important;
        box-shadow: 10px 10px 0px 0px #000000 !important;
        border-radius: 12px !important;
        padding: 2rem !important;
        color: #000000 !important;
        margin-bottom: 2rem;
    }

    /* FORCE BLACK TEXT in Sidebar and Cards */
    div[data-testid="stSidebar"] *, 
    .result-card *, 
    div.stForm * {
        color: #000000 !important;
    }

    /* Exception for colorful badges/headers if needed */
    .section-header {
        color: #ffffff !important; /* Keep headers white text */
    }

    /* Section Headers */
    .section-header {
        display: inline-flex;
        align-items: center;
        gap: 12px;
        padding: 8px 16px;
        background-color: #3F7D88;
        border: 2px solid black;
        box-shadow: 4px 4px 0px 0px black;
        border-radius: 8px;
        font-weight: 800;
        margin-bottom: 1.5rem;
        text-transform: uppercase;
    }

    /* BUTTON STYLING: White Background, Black Text for Max Visibility */
    .stButton > button {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 4px solid #000000 !important;
        border-radius: 12px !important;
        font-weight: 900 !important;
        text-transform: uppercase;
        padding: 1rem 1.5rem !important;
        transition: all 0.1s ease-in-out;
        box-shadow: 6px 6px 0px 0px #C05640 !important;
        width: 100%; 
        font-size: 1.2rem !important;
        display: block !important;
    }
    
    .stButton > button:hover {
        transform: translate(2px, 2px);
        box-shadow: 2px 2px 0px 0px #000000 !important;
        background-color: #f0f0f0 !important;
        color: #000000 !important;
    }
    
    .stButton > button p {
        color: #000000 !important; /* Force paragraph text inside buttons to black */
    }

    /* Form Inputs */
    input, select, textarea, div[data-testid*="stSelectbox"] {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 2px solid #000000 !important;
    }

    /* Wellness Score Box */
    .score-box {
        background-color: #ffffff;
        border: 6px solid #000000;
        box-shadow: 12px 12px 0px 0px #EAB308;
        padding: 2rem;
        text-align: center;
        margin-top: 2rem;
    }
    
    .score-box h1, .score-box h2 {
        color: #000000 !important;
    }

    /* AI Tools Header */
    .tool-header {
        font-weight: 900;
        font-size: 2rem;
        color: #ffffff !important;
        text-shadow: 3px 3px 0px #000000;
        margin-top: 3rem;
        margin-bottom: 1.5rem;
        text-align: center;
        text-transform: uppercase;
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
        st.error(f"Error loading model: {e}")
        return None

model = load_ml_model()

# --- GEMINI API CALL HANDLER ---
def call_gemini(prompt, is_json=True, max_retries=5):
    if not API_KEY:
        st.error("Gemini API key is missing. AI features disabled.")
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
    'Age', 'Gender', 'Academic_Level', 'Avg_Daily_Usage_Hours', 'Affects_Academic_Performance', 'Sleep_Hours_Per_Night', 'Conflicts_Over_Social_Media', 'Addicted_Score', 'Most_Used_Platform_Facebook', 'Most_Used_Platform_Instagram', 'Most_Used_Platform_KakaoTalk', 'Most_Used_Platform_LINE', 'Most_Used_Platform_LinkedIn', 'Most_Used_Platform_Snapchat', 'Most_Used_Platform_TikTok', 'Most_Used_Platform_Twitter', 'Most_Used_Platform_VKontakte', 'Most_Used_Platform_WeChat', 'Most_Used_Platform_WhatsApp', 'Most_Used_Platform_YouTube', 'Relationship_Status_Complicated', 'Relationship_Status_In Relationship', 'Relationship_Status_Single'
]

# --- UI LOGIC ---
st.markdown('<h1 style="color:white !important; text-align:center; font-size: 4rem; text-shadow: 4px 4px 0px black;">SOCIAL IMPACT</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:white !important; text-align:center; margin-bottom: 2rem; font-weight: bold;">Help us understand the digital footprint on your daily life.</p>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="section-header">👤 Profile</div>', unsafe_allow_html=True)
    age = st.number_input("Age", 10, 100, 20)
    gender = st.selectbox("Gender", ["Male", "Female"])
    academic_level = st.selectbox("Academic Level", ["High School", "Undergraduate", "Graduate"])
    
    st.markdown('<div class="section-header" style="background:#C05640;">📱 Usage</div>', unsafe_allow_html=True)
    avg_daily_usage = st.number_input("Daily Hours", 0.0, 24.0, 4.0, 0.5)
    platform = st.selectbox("Main Platform", ["TikTok", "YouTube", "Instagram", "Twitter", "Facebook", "Snapchat", "LINE", "KakaoTalk", "WhatsApp", "WeChat", "LinkedIn", "VKontakte"])
    addiction = st.slider("Addiction Score", 1, 10, 5)
    
    st.markdown('<div class="section-header" style="background:#EAB308;">❤️ Health</div>', unsafe_allow_html=True)
    sleep = st.number_input("Sleep Hours", 0.0, 24.0, 7.0, 0.5)
    affects_perf = st.selectbox("Impacts Academics?", ["No", "Yes"])
    conflicts = st.number_input("Social Media Conflicts", 0, 10, 0)
    rel_status = st.selectbox("Status", ["Single", "In a relationship", "Married", "Divorced"])
    
    st.markdown("<br>", unsafe_allow_html=True)
    calculate_button = st.button("RUN ANALYSIS ➔")

# --- APP LOGIC ---
if calculate_button:
    if model is None: st.stop()
    st.session_state.ai_results = {} 

    input_df = pd.DataFrame(0, index=[0], columns=MODEL_COLUMNS)
    try:
        input_df['Gender'] = 1 if gender == "Female" else 0 
        input_df['Age'] = age
        input_df['Academic_Level'] = {"High School": 0, "Undergraduate": 1, "Graduate": 2}.get(academic_level, 0)
        input_df['Avg_Daily_Usage_Hours'] = avg_daily_usage
        input_df['Addicted_Score'] = addiction
        input_df['Conflicts_Over_Social_Media'] = conflicts
        input_df['Affects_Academic_Performance'] = 1 if affects_perf == "Yes" else 0
        if 'Sleep_Hours' in MODEL_COLUMNS: input_df['Sleep_Hours'] = sleep
        
        # One-Hot Encoding
        plat_col = f"Most_Used_Platform_{platform}"
        if plat_col in MODEL_COLUMNS: input_df[plat_col] = 1
        rel_col = f"Relationship_Status_{rel_status}"
        if rel_col in MODEL_COLUMNS: input_df[rel_col] = 1

        wellness_score = model.predict(input_df)[0]
        st.session_state['score'] = wellness_score
        st.session_state['user_data_ai'] = {"Age": age, "Hours": avg_daily_usage, "Platform": platform, "Addiction": addiction, "Sleep": sleep}
    except Exception as e:
        st.error(f"Prediction Error: {e}")

if 'score' in st.session_state:
    score = st.session_state['score']
    st.markdown(f"""
    <div class="score-box">
        <h2 style='margin-bottom:0; font-weight:900;'>WELLNESS SCORE</h2>
        <h1 style='font-size: 6rem; color: {'#C05640' if score < 5 else '#3F7D88'}; font-weight: 900;'>{score:.2f} / 10</h1>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="tool-header">✨ AI COMMAND CENTER</div>', unsafe_allow_html=True)
    data = st.session_state['user_data_ai']
    c1, c2, c3 = st.columns(3)
    
    with c1:
        if st.button("📊 ANALYSIS"):
            res = call_gemini(f"Analyze: {json.dumps(data)}. Return JSON: {{\"persona\": \"archetype\", \"analysis\": \"2 sentences\", \"tips\": [\"tip1\", \"tip2\"]}}")
            if res: st.session_state.ai_results['analysis'] = json.loads(res); st.rerun()
    with c2:
        if st.button("🕰️ TIME TRAVEL"):
            res = call_gemini(f"Letter from 2029 future self based on: {json.dumps(data)}. Max 100 words.", is_json=False)
            if res: st.session_state.ai_results['future'] = res; st.rerun()
    with c3:
        if st.button("📅 DETOX"):
            res = call_gemini(f"3-day detox plan for {data['Platform']} user. Return JSON: {{\"days\": [{{\"day\": \"1\", \"theme\": \"...\", \"tasks\": []}}]}}")
            if res: st.session_state.ai_results['detox'] = json.loads(res); st.rerun()

    # Enhanced display area for AI results with proper formatting
    if st.session_state.get('ai_results'):
        for key, value in st.session_state.ai_results.items():
            content = ""
            if isinstance(value, dict):
                # Handle formatted JSON responses
                if key == 'analysis':
                    content = f"<b>Persona:</b> {value.get('persona', '')}<br><br>{value.get('analysis', '')}<br><ul>"
                    for tip in value.get('tips', []):
                        content += f"<li>{tip}</li>"
                    content += "</ul>"
                elif key == 'detox':
                    for day in value.get('days', []):
                        content += f"<b>{day.get('day')}: {day.get('theme')}</b><ul>"
                        for task in day.get('tasks', []):
                            content += f"<li>{task}</li>"
                        content += "</ul>"
            else:
                content = f"<p>{value}</p>"

            st.markdown(f"""
            <div class="result-card" style="border-left: 15px solid black;">
                <h3 style="text-transform: uppercase; margin-bottom: 10px; border-bottom: 2px solid black;">{key.upper()}</h3>
                <div style="font-size: 1.1rem;">{content}</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")
st.caption("Mature Neo-Brutalist Dashboard • STEAM Fair Project")
