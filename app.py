import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

# Set page configurations
st.set_page_config(page_title="RetiScan Pro v4", layout="wide", initial_sidebar_state="collapsed")

# 1. CONSTANTS & METADATA CONFIGURATIONS FROM RETI85% CODE
IMG_SIZE = 224
NUM_CLASSES = 5

CLASS_NAMES = {
    0: "No DR (Healthy)",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR"
}

SEVERITY_COLOR = {
    "No DR (Healthy)": "#10b981",
    "Mild NPDR":        "#f59e0b",
    "Moderate NPDR":    "#3b82f6",
    "Severe NPDR":      "#f97316",
    "Proliferative DR": "#ef4444"
}

MANAGEMENT = {
    "No DR (Healthy)": "No immediate action required. Maintain standard annual diabetic eye screening.",
    "Mild NPDR":        "Optimise glycaemic & blood pressure control. Re-examine in 6–12 months.",
    "Moderate NPDR":    "Ophthalmology referral recommended within 3–6 months. Consider HbA1c review.",
    "Severe NPDR":      "Urgent ophthalmology referral within 2–4 weeks. High risk of progression to PDR.",
    "Proliferative DR": "Immediate ophthalmology referral. High risk of vitreous haemorrhage & blindness. Laser / anti-VEGF therapy indicated."
}

CLINICAL_INFO = {
    "No DR (Healthy)": "The retina shows no signs of diabetic damage. Blood vessels appear normal with no microaneurysms, haemorrhages, or exudates detected.",
    "Mild NPDR":        "Early-stage NPDR characterised by at least one microaneurysm — small balloon-like swellings in the blood vessel walls of the retina.",
    "Moderate NPDR":    "Moderate NPDR. Multiple microaneurysms, dot/blot haemorrhages, and/or hard exudates present. Some capillary non-perfusion may be present.",
    "Severe NPDR":      "Severe NPDR (4-2-1 rule): >20 intraretinal haemorrhages in all 4 quadrants, venous beading in ≥2 quadrants, or IRMA in ≥1 quadrant. No neovascularisation yet.",
    "Proliferative DR": "Proliferative DR — new abnormal blood vessels (neovascularisation) have grown on the retina or optic disc. Vitreous/pre-retinal haemorrhage may be present."
}

# 2. CACHE MODEL TO SPEED UP LOADING TIMES
@st.cache_resource
def load_retiscan_model():
    # Looks for your file in the same repository directory
    return tf.keras.models.load_model("retiscan_pro_v4_best.keras")

try:
    model = load_retiscan_model()
    model_loaded = True
except Exception as e:
    st.error(f"Could not load the model file. Please ensure 'retiscan_pro_v4_best.keras' is uploaded to the repository root. Error: {e}")
    model_loaded = False

# 3. GRAPHICAL USER INTERFACE
st.markdown("<h1 style='text-align: center; color: #34d399;'>🔬 RetiScan Pro v4</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #94a3b8; font-weight: normal; margin-bottom: 30px;'>Diabetic Retinopathy Diagnostic System</h3>", unsafe_allow_html=True)

if model_loaded:
    uploaded_file = st.file_uploader("Upload Retinal Fundus Photograph (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert file buffer to an openCV image format
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # 🎯 FIX: MATCH CORRECT INFERENCE SCALE (0.0 to 255.0 floats) [cite: 33]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
        img_tensor = tf.convert_to_tensor(img_resized, dtype=tf.float32)
        img_batch = tf.expand_dims(img_tensor, axis=0)

        # Run model evaluation
        with st.spinner("Analyzing Retinal Scan Matrix..."):
            probabilities = model.predict(img_batch, verbose=0)[0]
        
        pred_idx = int(np.argmax(probabilities))
        pred_name = CLASS_NAMES[pred_idx]
        confidence = probabilities[pred_idx] * 100
        accent_color = SEVERITY_COLOR[pred_name]

        # Structure Application Display Columns
        col1, col2, col3 = st.columns([1, 1.2, 1.2], gap="large")

        with col1:
            st.markdown(f"<div style='border: 3px solid {accent_color}; border-radius: 10px; overflow: hidden;'><img src='data:image/jpeg;base64,{cv2.imencode('.jpg', img_bgr)[1].tobytes().hex()}' style='width:100%; display:block;'/></div>", unsafe_allow_html=True)
            # Alternate simple streamlit image display fall-back option if hex-render fails:
            st.image(img_rgb, use_container_width=True, caption=uploaded_file.name)

        with col2:
            st.markdown(f"<span style='color: #94a3b8; font-size: 12px; text-transform: uppercase;'>Diagnosis</span>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='color: {accent_color}; margin-top: 0px; font-weight: 800;'>{pred_name}</h2>", unsafe_allow_html=True)
            st.markdown(f"**Confidence Matrix Rating:** `{confidence:.2f}%`", unsafe_allow_html=True)
            st.markdown(f"<p style='color: #cbd5e1; font-size: 14px; line-height: 1.6; margin-top: 15px;'>{CLINICAL_INFO[pred_name]}</p>", unsafe_allow_html=True)

        with col3:
            st.markdown(f"<span style='color: #94a3b8; font-size: 12px; text-transform: uppercase;'>Grade Probabilities</span>", unsafe_allow_html=True)
            for i in range(NUM_CLASSES):
                cname = CLASS_NAMES[i]
                pct = probabilities[i]
                st.text(f"{cname} ({pct*100:.1f}%)")
                st.progress(float(pct))

        # Bottom Management Block
        st.markdown("<br>", unsafe_allow_html=True)
        st.info(f"📋 **Clinical Management Directive for {pred_name}:**\n\n{MANAGEMENT[pred_name]}")
        
        st.markdown("<hr style='border-color: #334155;'>", unsafe_allow_html=True)
        st.markdown("<p style='color: #64748b; font-size: 11px; text-align: center;'>⚠️ This tool is intended for research and screening assistance only. All diagnoses must be confirmed by a qualified ophthalmologist.</p>", unsafe_allow_html=True)
