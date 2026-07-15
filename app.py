import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytz
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore

# =====================================================================
# 1. PAGE CONFIGURATION & THEME CUSTOMIZATION
# =====================================================================
st.set_page_config(
    page_title="RetiScan Pro v5 Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS matching your dark dashboard theme
st.markdown("""
    <style>
    .main {
        background-color: #0d1117;
    }
    div[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    .stExpander {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 6px;
    }
    </style>
""", unsafe_allow_html=True)

# =====================================================================
# 2. SECURE FIREBASE CONNECTION
# =====================================================================
if not firebase_admin._apps:
    try:
        # Pull credentials securely from your Streamlit Secrets TOML
        cert_dict = dict(st.secrets["firebase"])
        cred = credentials.Certificate(cert_dict)
        firebase_admin.initialize_app(cred)
    except Exception as e:
        st.error(f"Error initializing Firebase. Please check your secrets configuration. Details: {e}")

db = firestore.client()

# =====================================================================
# 3. CLOUD DATABASE HELPER FUNCTIONS
# =====================================================================
def load_patient_history(p_id):
    """Fetches a specific patient's history from Firestore database."""
    try:
        doc_ref = db.collection('patients').document(p_id)
        doc = doc_ref.get()
        if doc.exists:
            return doc.to_dict().get('visits', [])
    except Exception as e:
        st.error(f"Error loading historical records: {e}")
    return []

def save_patient_record(p_id, diagnosis, confidence, attention_index):
    """Saves visit details to Firestore and returns the updated log."""
    # Strict Timezone Alignment (Corrects cloud server UTC to Local IST)
    ist_timezone = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist_timezone).strftime("%Y-%m-%d %H:%M")
    
    new_entry = {
        "timestamp": current_time,
        "diagnosis": diagnosis,
        "confidence": round(float(confidence), 2),
        "attention_index": round(float(attention_index), 2)
    }
    
    try:
        doc_ref = db.collection('patients').document(p_id)
        doc = doc_ref.get()
        
        if doc.exists:
            visits = doc.to_dict().get('visits', [])
            visits.append(new_entry)
            doc_ref.update({'visits': visits})
        else:
            visits = [new_entry]
            doc_ref.set({'visits': visits})
            
        return visits
    except Exception as e:
        st.error(f"Database write failure: {e}")
        return [new_entry]

# =====================================================================
# 4. CLINICAL CONTROL HUB (SIDEBAR)
# =====================================================================
st.sidebar.markdown("<h2 style='color: #8b949e; font-weight: 300;'>🩺 Clinical Control Hub</h2>", unsafe_allow_html=True)

patient_id = st.sidebar.text_input("👤 Patient ID Key", value="PATIENT-200").strip()
uploaded_file = st.sidebar.file_uploader("Upload Retinal Record Asset", type=["png", "jpg", "jpeg"])

# =====================================================================
# 5. MAIN DASHBOARD CONTENT
# =====================================================================
st.markdown("<h1 style='text-align: center; color: #58a6ff;'>🔬 RetiScan Pro v5 Dashboard</h1>", unsafe_allow_html=True)

if uploaded_file is not None:
    # -----------------------------------------------------------------
    # PLACEHOLDER FOR YOUR DEEP LEARNING MODEL / PROCESSING PIPELINE
    # -----------------------------------------------------------------
    # NOTE: Replace these placeholder processing blocks with your existing CV & model logic
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_rgb = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    
    # Mock output values (Replace with your model prediction calls)
    final_diagnosis = "No DR"  # Example: "No DR", "Mild NPDR", "Severe NPDR"
    confidence_score = 98.7
    attention_index = 0.0     # Quantitative metric calculation
    
    # Mocked secondary images (Replace with your actual pipeline outputs)
    gradcam_rgb = img_rgb.copy()       # Heatmap output
    vessel_map = img_rgb.copy()        # Vessel Segmentation map
    boundary_rgb = img_rgb.copy()      # ROI Bounding box output
    # -----------------------------------------------------------------
    
    # Trigger Firestore DB Transaction
    record_logs = save_patient_record(patient_id, final_diagnosis, confidence_score, attention_index)
    
    st.markdown("---")
    
    # =====================================================================
    # 6. ROW 1: PRIMARY DIAGNOSTIC VISUALS (COMPACT GRID SYSTEM)
    # =====================================================================
    st.markdown("<h4 style='color: #8b949e; font-size:13px; text-transform:uppercase; letter-spacing:1px; margin-bottom:15px;'>Primary Diagnostic Trackers</h4>", unsafe_allow_html=True)
    
    # Rebuilt into 3 columns (Wider layout, easier to analyze)
    img_col1, img_col2, img_col3 = st.columns(3, gap="large")
    
    with img_col1:
        st.markdown("<span style='color: #8b949e; font-size: 11px; text-transform: uppercase; font-weight:600;'>I. Base Input</span>", unsafe_allow_html=True)
        st.image(img_rgb, use_container_width=True)
        
    with img_col2:
        st.markdown("<span style='color: #8b949e; font-size: 11px; text-transform: uppercase; font-weight:600;'>II. Grad-CAM Path Map</span>", unsafe_allow_html=True)
        st.image(gradcam_rgb, use_container_width=True)
        
    with img_col3:
        st.markdown("<span style='color: #38bdf8; font-size: 11px; text-transform: uppercase; font-weight:600;'>III. Diagnostics Forecast (Longitudinal)</span>", unsafe_allow_html=True)
        
        # Chronological Graph Generation
        fig, ax = plt.subplots(figsize=(4, 3.5))
        
        visit_stamps = [r["timestamp"].split(" ")[0] for r in record_logs]
        attention_indices = [r["attention_index"] for r in record_logs]
        x_indices = np.arange(len(record_logs))
        
        ax.plot(x_indices, attention_indices, marker='o', color='#38bdf8', linewidth=2.5, label='Historical')
        
        if len(record_logs) >= 2:
            slope, intercept = np.polyfit(x_indices, attention_indices, 1)
            next_x = len(record_logs)
            next_y_pred = max(0.0, min(100.0, slope * next_x + intercept))
            
            forecast_x = [x_indices[-1], next_x]
            forecast_y = [attention_indices[-1], next_y_pred]
            ax.plot(forecast_x, forecast_y, linestyle='--', color='#ef4444', linewidth=2, marker='x', label='Forecast')
            ax.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#8b949e', fontsize=7)
        
        ax.set_xticks(range(len(record_logs) + (1 if len(record_logs) >= 2 else 0)))
        extended_stamps = visit_stamps + ["(Next)"] if len(record_logs) >= 2 else visit_stamps
        ax.set_xticklabels(extended_stamps, rotation=25, ha='right', fontsize=8)
        ax.set_ylim(-5, 105)
        ax.grid(True, linestyle='--', alpha=0.4)
        
        fig.patch.set_facecolor('#161b22')
        ax.set_facecolor('#0d1117')
        ax.spines['bottom'].set_color('#8b949e')
        ax.spines['left'].set_color('#8b949e')
        ax.tick_params(colors='#8b949e', labelsize=8)
        
        st.pyplot(fig)
        plt.close(fig)

    # =====================================================================
    # 7. ROW 2: ON-DEMAND EXPANDER FOR SECONDARY VISUALS
    # =====================================================================
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🔍 View Advanced Technical Layers (Vascular & AI ROI Focus)"):
        adv_col1, adv_col2 = st.columns(2, gap="medium")
        with adv_col1:
            st.markdown("<span style='color: #8b949e; font-size: 11px; text-transform: uppercase; font-weight:600;'>Vascular Topology Map</span>", unsafe_allow_html=True)
            st.image(vessel_map, use_container_width=True)
        with adv_col2:
            st.markdown(f"<span style='color: #00FFFF; font-size: 11px; text-transform: uppercase; font-weight:600;'>AI ROI Bounding Boxes ({attention_index:.1f} Index)</span>", unsafe_allow_html=True)
            st.image(boundary_rgb, use_container_width=True)

    # =====================================================================
    # 8. ROW 3: CLASSIFICATION & METRICS SUMMARY
    # =====================================================================
    st.markdown("<br><h4 style='color: #8b949e; font-size:13px; text-transform:uppercase; letter-spacing:1px; margin-bottom:15px;'>Classification & Metrics Summary</h4>", unsafe_allow_html=True)
    
    summary_col1, summary_col2 = st.columns(2, gap="large")
    
    with summary_col1:
        st.markdown(f"""
            <div style="background-color: #161b22; padding: 20px; border-radius: 8px; border: 1px solid #30363d;">
                <p style="color: #8b949e; font-size: 11px; text-transform: uppercase; margin: 0;">Diagnostic Verdict</p>
                <h1 style="color: #2ea44f; margin: 10px 0 0 0; font-size: 38px;">{final_diagnosis}</h1>
                <p style="color: #8b949e; font-size: 12px; margin-top: 10px;">Confidence Score: <b>{confidence_score}%</b> | Attention Intensity Score: <b>{attention_index}%</b></p>
            </div>
        """, unsafe_allow_html=True)
        
    with summary_col2:
        st.markdown("""
            <div style="background-color: #161b22; padding: 20px; border-radius: 8px; border: 1px solid #30363d; height: 100%;">
                <p style="color: #8b949e; font-size: 11px; text-transform: uppercase; margin: 0 0 10px 0;">Grade Probabilities Matrix</p>
                <div style="margin-bottom: 8px;">
                    <span style="color: #8b949e; font-size: 12px;">No DR</span>
                    <div style="background-color: #30363d; border-radius: 4px; height: 10px;">
                        <div style="background-color: #2ea44f; width: 98%; height: 100%; border-radius: 4px;"></div>
                    </div>
                </div>
                <div style="margin-bottom: 8px;">
                    <span style="color: #8b949e; font-size: 12px;">Mild NPDR</span>
                    <div style="background-color: #30363d; border-radius: 4px; height: 10px;">
                        <div style="background-color: #e3b341; width: 1.5%; height: 100%; border-radius: 4px;"></div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

else:
    # Fallback view when no patient scan is uploaded
    st.info("👈 Please enter a Patient ID and upload a retinal image in the Sidebar to begin clinical analysis.")
