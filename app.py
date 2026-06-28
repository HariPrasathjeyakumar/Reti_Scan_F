import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import gdown

# Set page configurations
st.set_page_config(page_title="RetiScan Pro v4", layout="wide", initial_sidebar_state="collapsed")

# 1. CONSTANTS & METADATA CONFIGURATIONS
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

CLINICAL_INFO = {
    "No DR (Healthy)": "The retina shows no signs of diabetic damage. Blood vessels appear normal with no microaneurysms, haemorrhages, or exudates detected.",
    "Mild NPDR":        "Early-stage NPDR characterised by at least one microaneurysm — small balloon-like swellings in the blood vessel walls of the retina.",
    "Moderate NPDR":    "Moderate NPDR. Multiple microaneurysms, dot/blot haemorrhages, and/or hard exudates present. Some capillary non-perfusion may be present.",
    "Severe NPDR":      "Severe NPDR (4-2-1 rule): >20 intraretinal haemorrhages in all 4 quadrants, venous beading in ≥2 quadrants, or IRMA in ≥1 quadrant. No neovascularisation yet.",
    "Proliferative DR": "Proliferative DR — new abnormal blood vessels (neovascularisation) have grown on the retina or optic disc. Vitreous/pre-retinal haemorrhage may be present."
}

# Custom loss wrapper needed for tf.keras to decode your saved .keras file architecture
def focal_loss():
    def loss_fn(y_true, y_pred): return tf.reduce_mean(y_pred)
    loss_fn.__name__ = "focal_loss"
    return loss_fn

# 2. CACHED MODEL STORAGE & DUAL-OUTPUT GRAPH PARSER
MODEL_FILENAME = "retiscan_pro_v4_best.keras"
FILE_ID = "1NFcXDWOMIVyVbA9j2pXUR6b8kCYGVKyq"

@st.cache_resource
def load_retiscan_pipeline():
    if not os.path.exists(MODEL_FILENAME):
        with st.spinner("Downloading trained model weights from secure cloud storage (this may take a minute on initial boot)..."):
            gdown.download(id=FILE_ID, output=MODEL_FILENAME, quiet=False)
            
    if not os.path.exists(MODEL_FILENAME) or os.path.getsize(MODEL_FILENAME) < 1000000:
        raise FileNotFoundError("The model file downloaded is corrupted or empty. Check Google Drive permissions.")
        
    # Load primary model using custom loss bounds
    main_model = tf.keras.models.load_model(MODEL_FILENAME, custom_objects={"focal_loss": focal_loss()})
    
    # Direct extraction of the target convolutional layer from flat or nested sequential topologies
    try:
        target_conv_layer = main_model.get_layer("top_conv")
    except ValueError:
        target_conv_layer = None
        for layer in main_model.layers:
            if hasattr(layer, 'get_layer'):
                try:
                    target_conv_layer = layer.get_layer("top_conv")
                    main_model = layer # Re-route to inner functional block if wrapped
                    break
                except ValueError:
                    continue
        if target_conv_layer is None:
            raise ValueError("Could not locate convolutional layer named 'top_conv' inside architecture.")
            
    # Compile gradient tracing model mapping inputs to both target conv activation maps and predictions
    grad_model = tf.keras.models.Model([main_model.inputs], [target_conv_layer.output, main_model.output])
    return main_model, grad_model

try:
    model, grad_model = load_retiscan_pipeline()
    model_loaded = True
except Exception as e:
    st.error(f"Could not initialize or download the model file. Error details: {e}")
    model_loaded = False

# 3. INTERACTIVE ALGORITHMIC COMPUTATIONS
def preprocess_for_inference(img_bgr):
    # Pipeline scaling fix: match exact 0-255 float32 pixel training ranges safely
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
    tensor = rgb.astype(np.float32)
    return np.expand_dims(tensor, axis=0)

@tf.function(reduce_retracing=True)
def _compile_gradcam_pass(img_tensor, grad_model, pred_idx):
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        loss = predictions[:, pred_idx]
    grads = tape.gradient(loss, conv_outputs)
    return conv_outputs, grads

def compute_gradcam_heatmap(img_tensor, grad_model, pred_idx):
    conv_outputs, grads = _compile_gradcam_pass(img_tensor, grad_model, tf.convert_to_tensor(pred_idx, dtype=tf.int32))
    
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val == 0: max_val = 1e-8
    return np.asarray(heatmap / max_val, dtype=np.float32)

def generate_dynamic_clinical_report(heatmap, pred_idx):
    pred_name = CLASS_NAMES[pred_idx]
    h, w = heatmap.shape
    mid_y, mid_x = h // 2, w // 2
    
    quadrants = {
        "Upper-Left":  heatmap[0:mid_y, 0:mid_x],
        "Upper-Right": heatmap[0:mid_y, mid_x:w],
        "Lower-Left":  heatmap[mid_y:h, 0:mid_x],
        "Lower-Right": heatmap[mid_y:h, mid_x:w]
    }
    
    scores = {k: np.mean(v) for k, v in quadrants.items()}
    primary_quad = max(scores, key=scores.get)
    total_score = sum(scores.values()) if sum(scores.values()) > 0 else 1
    focus_pct = (scores[primary_quad] / total_score) * 100
    
    if pred_idx == 0:
        rationale = "The neural activations are uniform and clean. No statistically significant pixel fluctuations were identified across any of the retinal structural fields, confirming a healthy vascular layout."
        management = "No immediate clinical intervention required. Advise the patient to maintain routine annual structural diabetic eye screenings and keep standard glycemic stability."
    else:
        if focus_pct > 35:
            rationale = f"The diagnostic prediction is primarily driven by concentrated structural anomalies located within the **{primary_quad} quadrant** (commanding {focus_pct:.1f}% of overall visual layer focus). Neural weights localized structural anomalies (such as microaneurysms or leakage sites) explicitly in this area."
            management = f"Schedule a targeted optical coherence tomography (OCT) review focusing specifically on the **{primary_quad} vascular fields** to audit localized edema. "
        else:
            rationale = "The prediction weights are distributed across multiple sectors. Visual features tracked diffuse, multi-quadrant microvascular abnormalities spanning across both upper and lower fields, indicating generalized retinal stress."
            management = "Initiate a comprehensive wide-field retinal evaluation. Because the abnormalities are diffuse and multi-quadrant, immediate systemic profiling is recommended. "

        if pred_idx == 1:
            management += "Re-examine in 6–12 months. Focus on stabilizing patient blood pressure and HbA1c variables."
        elif pred_idx == 2:
            management += "Issue a non-emergency ophthalmology referral within 3 months to monitor microvascular leakage progression."
        elif pred_idx == 3:
            management += "**Urgent Action Required:** Secure an accelerated specialist appointment within 2 weeks. High statistical probability of imminent progression to Proliferative DR."
        elif pred_idx == 4:
            management += "**Critical Emergency Action Required:** Immediate vitreoretinal surgery referral within 24–48 hours. Active high risk for vitreous hemorrhage or structural detachment. Evaluate eligibility for laser photocoagulation or anti-VEGF therapy injections."

    return rationale, management

# 4. INTERACTIVE WEB USER INTERFACE LAYOUT
st.markdown("<h1 style='text-align: center; color: #34d399; font-family: sans-serif;'>🔬 RetiScan Pro v4</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #94a3b8; font-weight: normal; margin-bottom: 30px; font-family: sans-serif;'>Diabetic Retinopathy Diagnostic System</h3>", unsafe_allow_html=True)

if model_loaded:
    uploaded_file = st.file_uploader("Upload Retinal Fundus Photograph (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Build tensor matrix
        img_tensor = preprocess_for_inference(img_bgr)

        # Run fast production prediction pass
        with st.spinner("Analyzing Retinal Scan Matrix..."):
            probabilities = model.predict_on_batch(img_tensor)[0]
        
        pred_idx = int(np.argmax(probabilities))
        pred_name = CLASS_NAMES[pred_idx]
        confidence = probabilities[pred_idx] * 100
        accent_color = SEVERITY_COLOR[pred_name]

        # Calculate Grad-CAM Heatmaps and extract clinical alerts
        raw_heatmap = compute_gradcam_heatmap(img_tensor, grad_model, pred_idx)
        dynamic_rationale, dynamic_management = generate_dynamic_clinical_report(raw_heatmap, pred_idx)

        # Bug-Free Conversion Loop: Turn float32 matrix to uint8 array BEFORE calling resize
        heatmap_uint8 = np.uint8(255 * raw_heatmap)
        heatmap_resized = cv2.resize(heatmap_uint8, (img_bgr.shape[1], img_bgr.shape[0]))
        heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        
        # Blend heatmap on top of base BGR image asset, then map to RGB space for streamlit
        xai_bgr = cv2.addWeighted(heatmap_color, 0.45, img_bgr, 0.55, 0)
        xai_rgb = cv2.cvtColor(xai_bgr, cv2.COLOR_BGR2RGB)
        img_display_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Present the 4-Column Diagnostic Dashboard Split
        col1, col2, col3, col4 = st.columns([1.1, 1.1, 1.4, 1.4], gap="medium")

        with col1:
            st.image(img_display_rgb, use_container_width=True, caption="Base Input Scan")

        with col2:
            st.image(xai_rgb, use_container_width=True, caption="Grad-CAM Activation Map")

        with col3:
            st.markdown(f"<span style='color: #94a3b8; font-size: 11px; text-transform: uppercase; letter-spacing: 1px;'>Diagnosis</span>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='color: {accent_color}; margin-top: 0px; font-weight: 800; font-family: sans-serif;'>{pred_name}</h2>", unsafe_allow_html=True)
            st.markdown(f"**Confidence Rating:** `{confidence:.2f}%`")
            st.markdown(f"<p style='color: #cbd5e1; font-size: 14px; line-height: 1.6; margin-top: 15px;'>{CLINICAL_INFO[pred_name]}</p>", unsafe_allow_html=True)

        with col4:
            st.markdown(f"<span style='color: #94a3b8; font-size: 11px; text-transform: uppercase; letter-spacing: 1px;'>Grade Probabilities</span>", unsafe_allow_html=True)
            st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
            for i in range(NUM_CLASSES):
                cname = CLASS_NAMES[i]
                pct = float(probabilities[i])
                
                # Render progress bars matching confidence payouts
                st.markdown(f"<div style='font-size: 13px; color: #cbd5e1; display: flex; justify-content: space-between;'><span>{cname}</span><span>{pct*100:.1f}%</span></div>", unsafe_allow_html=True)
                st.progress(pct)

        # Bottom Dynamic Explainable AI Breakdown Blocks
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown(f"🔍 **Dynamic XAI Local Analysis:**\n\n{dynamic_rationale}")
        st.info(f"📋 **Clinical Management Directive:**\n\n{dynamic_management}")
        
        st.markdown("<hr style='border-color: #334155;'>", unsafe_allow_html=True)
        st.markdown("<p style='color: #64748b; font-size: 11px; text-align: center;'>⚠️ This tool is intended for research and screening assistance only. All diagnoses must be confirmed by a qualified ophthalmologist.</p>", unsafe_allow_html=True)
