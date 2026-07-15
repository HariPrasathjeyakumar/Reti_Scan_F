import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import gdown
import json
import matplotlib.pyplot as plt
import pytz
from datetime import datetime
from fpdf import FPDF
from skimage.filters import frangi

# Professional UI Config - Wide Clinical Workspace
st.set_page_config(
    page_title="RetiScan Core OS", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Premium Midnight Steel EMR Styling Theme Hook
st.markdown("""
    <style>
    /* Main Background adjustments */
    .main {
        background-color: #0b0f17;
        color: #e2e8f0;
    }
    
    /* Control Sidebar Styling */
    div[data-testid="stSidebar"] {
        background-color: #131924;
        border-right: 1px solid #1e293b;
    }
    
    /* Metrics panel improvements */
    div[data-testid="stMetricValue"] {
        font-size: 24px !important;
        font-weight: 600 !important;
        color: #38bdf8 !important;
    }
    
    /* Global Card Architecture styling */
    .clinical-card {
        background-color: #131924;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    .stProgress > div > div > div > div {
        background-color: #38bdf8;
    }
    </style>
""", unsafe_allow_html=True)

# =====================================================================
#  1. CONSTANTS & SYSTEM CONFIGURATION
# =====================================================================
IMG_SIZE = 224
NUM_CLASSES = 5
MODEL_FILENAME = "retiscan_pro_v5_best.keras"
FILE_ID = "1NFcXDWOMIVyVbA9j2pXUR6b8kCYGVKyq"
HISTORY_FILE = "patient_history.json"

CLASS_NAMES = {
    0: "No DR", 1: "Mild NPDR", 2: "Moderate NPDR", 3: "Severe NPDR", 4: "Proliferative DR"
}

SEVERITY_COLOR = {
    "No DR": "#10b981", "Mild NPDR": "#f59e0b", "Moderate NPDR": "#38bdf8", "Severe NPDR": "#e11d48", "Proliferative DR": "#f43f5e"
}

CLASS_DESCRIPTIONS = {
    0: "Normal clinical presentation. Retina matrix displays healthy vascular infrastructure without localized abnormalities.",
    1: "Early stage microvascular microaneurysms detected within peripheral capillary distributions.",
    2: "Moderate intraretinal microvascular variations noted alongside localized microhemorrhages.",
    3: "Severe structural compromise observed across multiple retinal quadrants.",
    4: "Advanced proliferative disease indicating active neovascularization loops."
}

CLINICAL_DIRECTIVES = {
    0: "Routine screening tracking check in 12 months. Manage standard systemic vascular health.",
    1: "Schedule non-urgent OCT scan. Monitor macular boundaries closely in 6 months.",
    2: "Referral to medical retina specialist. Address underlying systemic markers.",
    3: "Urgent specialist evaluation recommended. Prepare for fast-track PRP profiling.",
    4: "CRITICAL ALERT: Emergency vitreo-retinal consultation required for immediate intervention mapping."
}

def focal_loss():
    def loss_fn(y_true, y_pred): return tf.reduce_mean(y_pred)
    loss_fn.__name__ = "focal_loss"
    return loss_fn

# =====================================================================
#  2. MODEL LOADING & RECORDS DATABASE PIPELINE
# =====================================================================
@st.cache_resource
def load_retiscan_pipeline():
    if not os.path.exists(MODEL_FILENAME):
        with st.spinner("Synchronizing diagnostic system weights..."):
            gdown.download(id=FILE_ID, output=MODEL_FILENAME, quiet=False)
    
    main_model = tf.keras.models.load_model(MODEL_FILENAME, custom_objects={"focal_loss": focal_loss()})
    
    conv_layer_name = None
    for layer in reversed(main_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D) or 'top_conv' in layer.name or 'block7' in layer.name:
            conv_layer_name = layer.name
            break
    if not conv_layer_name: conv_layer_name = "top_conv"
    
    grad_model = tf.keras.models.Model([main_model.inputs], [main_model.get_layer(conv_layer_name).output, main_model.output])
    return main_model, grad_model

try:
    model, grad_model = load_retiscan_pipeline()
    model_loaded = True
except Exception as e:
    st.error(f"Initialization Failed: {e}")
    model_loaded = False

def load_patient_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f: return json.load(f)
        except: return {}
    return {}

def save_patient_record(p_id, diagnosis, confidence, attention_index):
    history = load_patient_history()
    if p_id not in history: history[p_id] = []
    
    ist_tz = pytz.timezone('Asia/Kolkata')
    current_time_ist = datetime.now(ist_tz).strftime("%Y-%m-%d %H:%M")
    
    new_entry = {
        "timestamp": current_time_ist, "diagnosis": diagnosis,
        "confidence": round(float(confidence), 2), "attention_index": round(float(attention_index), 2)
    }
    history[p_id].append(new_entry)
    with open(HISTORY_FILE, "w") as f: json.dump(history, f, indent=4)
    return history[p_id]

# =====================================================================
#  3. COMPUTER VISION & CORE SCREENING ENGINES
# =====================================================================
def preprocess_for_inference(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
    return np.expand_dims(rgb.astype(np.float32), axis=0)

def run_tta_ensemble_inference(model, base_tensor):
    pred_base = model.predict_on_batch(base_tensor)[0]
    pred_flipped = model.predict_on_batch(np.flip(base_tensor, axis=2))[0]
    pred_gamma = model.predict_on_batch(np.clip(base_tensor * 1.05, 0.0, 255.0))[0]
    
    fused_probs = (pred_base + pred_flipped + pred_gamma) / 3.0
    mean_variance = np.mean(np.var([pred_base, pred_flipped, pred_gamma], axis=0))
    consensus = "HIGH CONSENSUS" if mean_variance < 0.02 else "RE-VERIFICATION ADVISED"
    return fused_probs, consensus

def run_pre_computing_screening(img_bgr):
    h, w, _ = img_bgr.shape
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    if cv2.Laplacian(gray, cv2.CV_64F).var() < 45.0:
        return False, "Focal clarity limits violated (Blurry).", None, None, None
    if np.mean(gray) < 10:
        return False, "Exposure limit error (Asset too dark).", None, None, None
        
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return False, "Retinal geometry boundaries undefined.", None, None, None
    
    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < (h * w * 0.15):
        return False, "Insufficient fundus structure surface area.", None, None, None
        
    (x_c, y_c), radius = cv2.minEnclosingCircle(largest_contour)
    circle_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(circle_mask, (int(x_c), int(y_c)), int(radius * 0.8), 255, -1)
    mean_b, mean_g, mean_r, _ = cv2.mean(img_bgr, mask=circle_mask)
    
    if mean_b > 90 or mean_r < mean_g:
        return False, "Atypical physiological profile detected.", None, None, None
    return True, "PASSED", x_c, y_c, radius

def generate_vascular_map(img_bgr, x_c, y_c, radius):
    h, w, _ = img_bgr.shape
    enhanced_g = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(cv2.split(img_bgr)[1])
    vesselness = frangi(cv2.bitwise_not(enhanced_g), sigmas=range(1, 4, 1), black_ridges=False)
    vesselness_norm = cv2.normalize(vesselness, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (int(x_c), int(y_c)), int(radius * 0.92), 255, -1)
    return cv2.bitwise_and(vesselness_norm, mask)

def compute_diagnostic_graphs(img_tensor, grad_model, pred_idx, img_bgr, x_c, y_c, radius):
    h, w, _ = img_bgr.shape
    with tf.GradientTape() as tape:
        conv_outputs, model_predictions = grad_model(img_tensor)
        loss = model_predictions[:, pred_idx]
        
    grads = tape.gradient(loss, conv_outputs)
    heatmap = tf.squeeze(conv_outputs[0] @ tf.reduce_mean(grads, axis=(0, 1, 2))[..., tf.newaxis])
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    raw_heatmap = (heatmap / (max_val if max_val > 0 else 1e-8)).numpy()
    
    isolated_heatmap = cv2.bitwise_and(
        cv2.resize(np.uint8(255 * raw_heatmap), (w, h)),
        cv2.circle(np.zeros((h, w), dtype=np.uint8), (int(x_c), int(y_c)), int(radius * 0.80), 255, -1)
    )
        
    boundary_img = img_bgr.copy()
    if pred_idx == 0:
        attn_index = 0.0
    else:
        act_pixels = isolated_heatmap[isolated_heatmap > 0]
        attn_index = np.mean(act_pixels[act_pixels >= np.percentile(act_pixels, 90)]) / 255.0 * 100.0 if len(act_pixels) > 0 else 0.0
        
        _, binary_mask = cv2.threshold(isolated_heatmap, int(0.55 * (np.max(isolated_heatmap) if np.max(isolated_heatmap) > 0 else 1)), 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) > 25:
                cv2.drawContours(boundary_img, [c], -1, (0, 255, 255), 1)
                x, y, bw, bh = cv2.boundingRect(c)
                cv2.rectangle(boundary_img, (x, y), (x+bw, y+bh), (0, 140, 255), 2)
                
    quad_y, quad_x = int(y_c), int(x_c)
    quads = {
        "Upper-Left": isolated_heatmap[0:quad_y, 0:quad_x], "Upper-Right": isolated_heatmap[0:quad_y, quad_x:w],
        "Lower-Left": isolated_heatmap[quad_y:h, 0:quad_x], "Lower-Right": isolated_heatmap[quad_y:h, quad_x:w]
    }
    quad_sums = {k: np.sum(v) for k, v in quads.items()}
    tot_sum = np.sum(list(quad_sums.values()))
    dom_quad = max(quad_sums, key=quad_sums.get) if tot_sum > 0 else "Central"
    quad_pct = (quad_sums[dom_quad] / tot_sum * 100) if tot_sum > 0 else 0.0
    
    return isolated_heatmap, boundary_img, attn_index, dom_quad, quad_pct

def generate_clinical_pdf(p_id, verdict, conf, attn_idx, quad, quad_pct, directive, consensus):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 12, "RETISCAN DIAGNOSTIC REPORT", ln=True, align="L")
    pdf.ln(5)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 6, f"Patient Reference: {p_id}", ln=True)
    pdf.cell(0, 6, f"Pipeline State: {consensus}", ln=True)
    pdf.cell(0, 6, f"Calculated Severity Class: {verdict} ({conf:.1f}% Confidence)", ln=True)
    pdf.cell(0, 6, f"Primary Focus Cluster: {quad} Quadrant ({quad_pct:.1f}% Weight)", ln=True)
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "Management Framework Guideline:", ln=True)
    pdf.set_font("Helvetica", "I", 11)
    pdf.multi_cell(0, 6, directive)
    return bytes(pdf.output())

# =====================================================================
#  4. CLINICAL DASHBOARD LAYOUT & HEADLESS WORKSPACE
# =====================================================================
st.markdown("<div style='margin-top: 10px; margin-bottom: 25px;'><span style='background-color:#1e293b; color:#38bdf8; padding:5px 10px; border-radius:12px; font-size:11px; font-weight:700; letter-spacing:0.5px;'>CLINICAL INTERACTION PLATFORM</span><h1 style='color:#ffffff; margin:8px 0 0 0; font-weight:500; font-size:28px;'>RetiScan Core OS</h1></div>", unsafe_allow_html=True)

st.sidebar.markdown("<p style='color:#64748b; font-size:11px; font-weight:700; letter-spacing:1px; text-transform:uppercase; margin-bottom:15px;'>Workspace Navigation</p>", unsafe_allow_html=True)
patient_id = st.sidebar.text_input("👤 Unique Patient ID", value="REF-9021").strip().upper()

if model_loaded:
    uploaded_file = st.sidebar.file_uploader("Ingest Fundus Record Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        passed, msg, x_c, y_c, radius = run_pre_computing_screening(img_bgr)
        
        if not passed:
            st.markdown(f"<div style='background-color:#7f1d1d; border-left:4px solid #f43f5e; padding:15px; border-radius:4px; font-size:13.5px; color:#fecdd3; font-weight:500;'>⚠️ Operational Hazard: {msg}</div>", unsafe_allow_html=True)
        else:
            img_tensor = preprocess_for_inference(img_bgr)
            probabilities, consensus_status = run_tta_ensemble_inference(model, img_tensor)
            
            pred_idx = int(np.argmax(probabilities))
            pred_name = CLASS_NAMES[pred_idx]
            confidence = probabilities[pred_idx] * 100
            
            isolated_heatmap, boundary_img, attn_index, dom_quad, quad_pct = compute_diagnostic_graphs(
                img_tensor, grad_model, pred_idx, img_bgr, x_c, y_c, radius
            )
            vessel_map = generate_vascular_map(img_bgr, x_c, y_c, radius)
            record_logs = save_patient_record(patient_id, pred_name, confidence, attention_index=attn_index)
            
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            blend_rgb = cv2.cvtColor(cv2.addWeighted(cv2.applyColorMap(isolated_heatmap, cv2.COLORMAP_JET), 0.35, img_bgr, 0.65, 0), cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            boundary_rgb = cv2.cvtColor(boundary_img, cv2.COLOR_BGR2RGB)

            # --- SYSTEM STATUS SUMMARY METRICS ROW ---
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1: st.metric("Calculated Classification Target", pred_name)
            with m_col2: st.metric("Ensemble Confidence Core", f"{confidence:.1f}%")
            with m_col3: st.metric("Neural Target Density", f"{attn_index:.1f}%")
            with m_col4: st.metric("Operational Status", consensus_status)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # --- MAIN DOUBLE MODULE LAYOUT ---
            col_left, col_right = st.columns([7, 5], gap="large")
            
            with col_left:
                st.markdown("<p style='color:#64748b; font-size:11px; font-weight:700; text-transform:uppercase;'>Primary Imaging Modalities</p>", unsafe_allow_html=True)
                tab1, tab2, tab3 = st.tabs(["Raw Asset Focus", "Grad-CAM Localization Map", "Vascular Segmentation Matrix"])
                with tab1: st.image(img_rgb, use_container_width=True)
                with tab2: st.image(blend_rgb, use_container_width=True)
                with tab3: st.image(vessel_map, use_container_width=True, clamp=True)
                
                # Clinical Directives & Context box
                st.markdown(f"""
                    <div class="clinical-card" style="margin-top:20px;">
                        <span style="font-size:11px; color:#64748b; font-weight:700; text-transform:uppercase;">System Directives & Contextual Evaluation</span>
                        <h4 style="color:#ffffff; font-weight:500; margin:10px 0 5px 0; font-size:16px;">Diagnostic Summary Profile</h4>
                        <p style="color:#94a3b8; font-size:13.5px; line-height:1.5; margin-bottom:15px;">{CLASS_DESCRIPTIONS[pred_idx]}</p>
                        <div style="background-color:#1e293b; border-left:3px solid #38bdf8; padding:12px; border-radius:4px;">
                            <span style="font-size:10px; color:#38bdf8; font-weight:700; text-transform:uppercase; display:block; margin-bottom:3px;">Clinical Path Management Guideline</span>
                            <span style="font-size:13px; color:#e2e8f0; font-weight:500;">{CLINICAL_DIRECTIVES[pred_idx]}</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
            with col_right:
                st.markdown("<p style='color:#64748b; font-size:11px; font-weight:700; text-transform:uppercase;'>Numerical Analytics Matrix</p>", unsafe_allow_html=True)
                
                with st.container():
                    st.markdown("<div class='clinical-card'>", unsafe_allow_html=True)
                    st.markdown("<span style='font-size:11px; color:#64748b; font-weight:700; text-transform:uppercase; display:block; margin-bottom:15px;'>Probability Mapping Profiles</span>", unsafe_allow_html=True)
                    for i in range(NUM_CLASSES):
                        cname = CLASS_NAMES[i]
                        p_val = float(probabilities[i])
                        lbl_color = "#ffffff" if i == pred_idx else "#94a3b8"
                        st.markdown(f"<div style='display:flex; justify-content:between; font-size:12px; color:{lbl_color}; margin-bottom:2px; font-weight:500;'><span>{cname}</span><span style='margin-left:auto;'>{p_val*100:.1f}%</span></div>", unsafe_allow_html=True)
                        st.progress(p_val)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with st.container():
                    st.markdown("<div class='clinical-card'>", unsafe_allow_html=True)
                    st.markdown("<span style='font-size:11px; color:#64748b; font-weight:700; text-transform:uppercase; display:block; margin-bottom:10px;'>Longitudinal Structural Trend Analysis</span>", unsafe_allow_html=True)
                    
                    fig, ax = plt.subplots(figsize=(5, 2.8))
                    x_idx = np.arange(len(record_logs))
                    y_vals = [r["attention_index"] for r in record_logs]
                    
                    ax.plot(x_idx, y_vals, marker='o', color='#38bdf8', linewidth=2, label='Observed Index')
                    if len(record_logs) >= 2:
                        m, c = np.polyfit(x_idx, y_vals, 1)
                        ax.plot([x_idx[-1], len(record_logs)], [y_vals[-1], max(0.0, min(100.0, m * len(record_logs) + c))], linestyle='--', color='#f43f5e', label='Trend Forecast')
                        prog_txt = "⚠️ PROGRESSION RATIONALE DETECTED" if m > 1.5 else "STABLE DYNAMICS OBSERVED"
                    else:
                        prog_txt = "AWAITING SECURED BASELINE ENTRIES"
                        
                    ax.set_ylim(-10, 110)
                    ax.set_xticks(x_idx)
                    ax.set_xticklabels([r["timestamp"].split(" ")[0] for r in record_logs], rotation=15, color='#94a3b8', fontsize=8)
                    fig.patch.set_facecolor('#131924')
                    ax.set_facecolor('#0b0f17')
                    ax.spines['bottom'].set_color('#1e293b')
                    ax.spines['left'].set_color('#1e293b')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.tick_params(colors='#94a3b8', labelsize=8)
                    ax.grid(True, linestyle=':', alpha=0.15, color='#e2e8f0')
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    st.markdown(f"<div style='font-size:11px; text-align:center; color:#94a3b8; margin-top:5px; font-weight:600;'>PROGNOSTIC TREND: <span style='color:#38bdf8;'>{prog_txt}</span></div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

            # --- ADVANCED STRUCTURAL MAPPING BOUNDARIES ---
            with st.expander("🛠️ View Object Detection Boundary & Target Layers"):
                st.image(boundary_rgb, use_container_width=True)

            # --- HISTORICAL PATIENT FILE REGISTRATION ---
            st.markdown("<br><p style='color:#64748b; font-size:11px; font-weight:700; text-transform:uppercase;'>Verified Patient Tracking Logs</p>", unsafe_allow_html=True)
            tbl = '<table style="width:100%; border-collapse:collapse; font-size:13px; background-color:#131924; border:1px solid #1e293b; border-radius:6px; color:#e2e8f0;">'
            tbl += '<tr style="background-color:#0b0f17; border-bottom:2px solid #1e293b; color:#94a3b8;"><th style="padding:10px;">Visit ID</th><th style="padding:10px;">Date & Time (IST)</th><th style="padding:10px;">Diagnostic Verdict</th><th style="padding:10px;">Confidence</th><th style="padding:10px;">Target Density Index</th></tr>'
            for idx, r in enumerate(record_logs):
                tbl += f'<tr style="border-bottom:1px solid #1e293b;"><td style="padding:10px; font-weight:600;">#0{idx+1}</td><td style="padding:10px;">{r["timestamp"]}</td><td style="padding:10px;">{r["diagnosis"]}</td><td style="padding:10px;">{r["confidence"]}%</td><td style="padding:10px; color:#38bdf8; font-weight:600;">{r["attention_index"]:.1f}%</td></tr>'
            tbl += '</table>'
            st.markdown(tbl, unsafe_allow_html=True)
            
            # --- DOWNLOAD PIPELINE ---
            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button(
                label="📥 Generate Signed EMR Diagnostic Document Portfolio (PDF)",
                data=generate_clinical_pdf(patient_id, pred_name, confidence, attn_index, dom_quad, quad_pct, CLINICAL_DIRECTIVES[pred_idx], consensus_status),
                file_name=f"EMR_{patient_id}_{datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
else:
    st.info("💡 Awaiting Clinical Ingestion Setup: Complete structural fields inside control sidebar pipeline to start diagnostics tracking.")
