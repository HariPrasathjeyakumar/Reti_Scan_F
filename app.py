import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pytz
from datetime import datetime

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
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)

HISTORY_FILE = "patient_history.json"

# =====================================================================
# 2. LOCAL DATA STORAGE HELPER FUNCTIONS (WITH IST TIME CORRECTION)
# =====================================================================
def load_patient_history(p_id):
    """Loads historical records for a patient from a local JSON file."""
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r") as f:
            data = json.load(f)
            return data.get(p_id, [])
    except Exception:
        return []

def save_patient_record(p_id, diagnosis, confidence, attention_index):
    """Saves a new record to the local JSON file using Indian Standard Time (IST)."""
    # Load existing history
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                history = json.load(f)
        except Exception:
            history = {}
    else:
        history = {}

    if p_id not in history:
        history[p_id] = []
        
    # Correcting the clock: Force Indian Standard Time (IST) 
    ist_timezone = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist_timezone).strftime("%Y-%m-%d %H:%M")
    
    new_entry = {
        "timestamp": current_time,
        "diagnosis": diagnosis,
        "confidence": round(float(confidence), 2),
        "attention_index": round(float(attention_index), 2)
    }
    
    history[p_id].append(new_entry)
    
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)
        
    return history[p_id]

# =====================================================================
# 3. CLINICAL CONTROL HUB (SIDEBAR)
# =====================================================================
st.sidebar.markdown("<h2 style='color: #8b949e; font-weight: 300; margin-bottom: 20px;'>🩺 Clinical Control Hub</h2>", unsafe_allow_html=True)

patient_id = st.sidebar.text_input("👤 Patient ID Key", value="PATIENT-200").strip()
uploaded_file = st.sidebar.file_uploader("Upload Retinal Record Asset", type=["png", "jpg", "jpeg"])

# =====================================================================
# 4. MAIN DASHBOARD CONTENT
# =====================================================================
st.markdown("<h1 style='color: #58a6ff; font-weight: 400; margin-bottom: 20px;'>🔬 RetiScan Pro v5 Dashboard</h1>", unsafe_allow_html=True)

if uploaded_file is not None:
    # -----------------------------------------------------------------
    # PLACEHOLDER FOR DEEP LEARNING MODEL / PROCESSING PIPELINE
    # -----------------------------------------------------------------
    # Convert uploaded file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_rgb = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    
    # --- MOCK OUTCOMES (Plug in your real deep learning network variables here) ---
    final_diagnosis = "No DR"  
    confidence_score = 98.67
    attention_index = 0.0     
    
    # --- MOCK IMAGE PIPELINE OUTPUTS (Replace with your model's actual images) ---
    gradcam_rgb = img_rgb.copy()       # Heatmap matrix
    vessel_map = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY) # Vessel extraction simulation
    boundary_rgb = img_rgb.copy()      # Bounding boxes matrix
    # -----------------------------------------------------------------
    
    # Save current run data locally and pull patient diagnostic timeline logs
    record_logs = save_patient_record(patient_id, final_diagnosis, confidence_score, attention_index)
    
    st.markdown("---")
    
    # =====================================================================
    # 5. PRIMARY DIAGNOSTIC VISUALS (CLEAN 3-COLUMN STRUCTURE)
    # =====================================================================
    st.markdown("<h4 style='color: #8b949e; font-size:13px; text-transform:uppercase; letter-spacing:1px; margin-bottom:15px;'>Primary Diagnostic Trackers</h4>", unsafe_allow_html=True)
    
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
        
        # Apply forecasting linear projection if there's enough timeline history
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
        ax.grid(True, linestyle='--', alpha=0.2, color='#8b949e')
        
        # Match Dark theme styling parameters
        fig.patch.set_facecolor('#161b22')
        ax.set_facecolor('#0d1117')
        ax.spines['bottom'].set_color('#30363d')
        ax.spines['left'].set_color('#30363d')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(colors='#8b949e', labelsize=8)
        
        st.pyplot(fig)
        plt.close(fig)

    # =====================================================================
    # 6. ON-DEMAND EXPANDER PANEL (HIDES MESSY/SECONDARY IMAGES)
    # =====================================================================
    with st.expander("🔍 View Advanced Technical Layers (Vascular Topology & AI ROI Focus)"):
        adv_col1, adv_col2 = st.columns(2, gap="medium")
        with adv_col1:
            st.markdown("<span style='color: #8b949e; font-size: 11px; text-transform: uppercase; font-weight:600;'>Vascular Topology Map</span>", unsafe_allow_html=True)
            st.image(vessel_map, use_container_width=True, clamp=True)
        with adv_col2:
            st.markdown(f"<span style='color: #38bdf8; font-size: 11px; text-transform: uppercase; font-weight:600;'>AI ROI Bounding Boxes ({attention_index:.1f} Index)</span>", unsafe_allow_html=True)
            st.image(boundary_rgb, use_container_width=True)

    # =====================================================================
    # 7. CLASSIFICATION & METRICS SUMMARY CARD ENGINE
    # =====================================================================
    st.markdown("<br><h4 style='color: #8b949e; font-size:13px; text-transform:uppercase; letter-spacing:1px; margin-bottom:15px;'>Classification & Metrics Summary</h4>", unsafe_allow_html=True)
    
    summary_col1, summary_col2 = st.columns(2, gap="large")
    
    with summary_col1:
        st.markdown(f"""
            <div style="background-color: #161b22; padding: 22px; border-radius: 6px; border: 1px solid #30363d;">
                <p style="color: #8b949e; font-size: 11px; text-transform: uppercase; margin: 0; letter-spacing: 0.5px;">Diagnostic Verdict</p>
                <h1 style="color: #2ea44f; margin: 10px 0 5px 0; font-size: 40px; font-weight: 500;">{final_diagnosis}</h1>
                <p style="color: #58a6ff; font-size: 12px; margin-bottom: 12px; font-weight: 500;">Integrity Consensus Validation: HIGH CONSENSUS</p>
                <p style="color: #8b949e; font-size: 12px; margin: 0;">Confidence: <b>{confidence_score}%</b> | Attention Intensity Score: <b>{attention_index}%</b></p>
                <p style="color: #8b949e; font-size: 12px; margin-top: 8px; line-height: 1.4;">Normal healthy retina matrix. No microaneurysms, hemorrhages, or lipid exudates detected within the mapped vascular fields.</p>
            </div>
        """, unsafe_allow_html=True)
        
    with summary_col2:
        st.markdown("""
            <div style="background-color: #161b22; padding: 22px; border-radius: 6px; border: 1px solid #30363d; height: 100%;">
                <p style="color: #8b949e; font-size: 11px; text-transform: uppercase; margin: 0 0 15px 0; letter-spacing: 0.5px;">Grade Probabilities Matrix</p>
                <div style="margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
                        <span style="color: #c9d1d9; font-size: 12px;">No DR</span>
                        <span style="color: #2ea44f; font-size: 12px; font-weight: 600;">98.7%</span>
                    </div>
                    <div style="background-color: #30363d; border-radius: 4px; height: 8px;">
                        <div style="background-color: #2ea44f; width: 98.7%; height: 100%; border-radius: 4px;"></div>
                    </div>
                </div>
                <div style="margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
                        <span style="color: #c9d1d9; font-size: 12px;">Mild NPDR</span>
                        <span style="color: #8b949e; font-size: 12px;">0.4%</span>
                    </div>
                    <div style="background-color: #30363d; border-radius: 4px; height: 8px;">
                        <div style="background-color: #e3b341; width: 4%; height: 100%; border-radius: 4px;"></div>
                    </div>
                </div>
                <div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
                        <span style="color: #c9d1d9; font-size: 12px;">Moderate NPDR</span>
                        <span style="color: #8b949e; font-size: 12px;">0.2%</span>
                    </div>
                    <div style="background-color: #30363d; border-radius: 4px; height: 8px;">
                        <div style="background-color: #f778ba; width: 2%; height: 100%; border-radius: 4px;"></div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

else:
    # Warm welcome landing panel
    st.info("👈 Complete the entry inputs in the Clinical Control Hub sidebar to initialize data processing pathways.")
