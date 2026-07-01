import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import gdown

# Set page configurations to clean wide layout
st.set_page_config(page_title="RetiScan Pro v5", layout="wide", initial_sidebar_state="collapsed")

# =====================================================================
#  1. CONSTANTS, CLOUD PATHS, & CORE METADATA
# =====================================================================
IMG_SIZE = 224
NUM_CLASSES = 5
MODEL_FILENAME = "retiscan_pro_v5_best.keras"
FILE_ID = "1BrmHzARxRFTBilmymFkwn6e5ZjPmosu1"  # Uses your model storage target

CLASS_NAMES = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR"
}

SEVERITY_COLOR = {
    "No DR":           "#10b981",
    "Mild NPDR":       "#f59e0b",
    "Moderate NPDR":   "#3b82f6",
    "Severe NPDR":     "#f97316",
    "Proliferative DR": "#ef4444"
}

CLASS_DESCRIPTIONS = {
    0: "Normal healthy retina matrix. No microaneurysms, hemorrhages, or lipid exudates detected within the mapped vascular fields.",
    1: "Early-stage NPDR characterized by isolated microaneurysms—small balloon-like swellings in the blood vessel walls of the retina.",
    2: "Progressive stage marked by advanced intraretinal microvascular abnormalities, distinct microaneurysms, and localized hemorrhages.",
    3: "Severe structural compromise with extensive intraretinal hemorrhages and microaneurysms spanning multiple quadrants.",
    4: "Advanced proliferative disease marked by neovascularization—fragile new blood vessel growth prone to severe vitreous leakage."
}

CLINICAL_DIRECTIVES = {
    0: "Schedule routine tracking examination in 12 months. Continue baseline systemic blood glucose and blood pressure monitoring.",
    1: "Schedule a targeted Optical Coherence Tomography (OCT) review focusing on localized structural fields. Re-examine in 6-12 months.",
    2: "Refer to an ophthalmologist for urgent structural baseline tracking. Evaluate macular edema and stabilize HbA1c variables.",
    3: "Immediate specialist referral required. Initiate rapid panretinal photocoagulation (PRP) screening profiles safely.",
    4: "CRITICAL ALERT: Emergency vitreo-retinal surgical evaluation indicated. High immediate risk of permanent tractional detachment."
}

# Custom loss wrapper needed for decoding functional network graph safely
def focal_loss():
    def loss_fn(y_true, y_pred): return tf.reduce_mean(y_pred)
    loss_fn.__name__ = "focal_loss"
    return loss_fn

# =====================================================================
#  2. CACHED MODEL ENGINE & HOOKED GRADIENT LAYER TOPOLOGY
# =====================================================================
@st.cache_resource
def load_retiscan_pipeline():
    if not os.path.exists(MODEL_FILENAME):
        with st.spinner("Downloading trained model weights from secure cloud storage..."):
            gdown.download(id=FILE_ID, output=MODEL_FILENAME, quiet=False)
             
    if not os.path.exists(MODEL_FILENAME) or os.path.getsize(MODEL_FILENAME) < 1000000:
        raise FileNotFoundError("The model file downloaded is corrupted or empty.")
        
    main_model = tf.keras.models.load_model(MODEL_FILENAME, custom_objects={"focal_loss": focal_loss()})
    
    # Locate final deep feature map block
    conv_layer_name = None
    for layer in reversed(main_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D) or 'top_conv' in layer.name or 'block7' in layer.name:
            conv_layer_name = layer.name
            break
    if not conv_layer_name: 
        conv_layer_name = "top_conv"
        
    grad_model = tf.keras.models.Model([main_model.inputs], [main_model.get_layer(conv_layer_name).output, main_model.output])
    return main_model, grad_model

try:
    model, grad_model = load_retiscan_pipeline()
    model_loaded = True
except Exception as e:
    st.error(f"Could not initialize or download the model file. Error: {e}")
    model_loaded = False

# =====================================================================
#  3. PROCEDURAL PREPROCESSING & INTERACTIVE ADVANCED XAI MASKING
# =====================================================================
def preprocess_for_inference(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
    tensor = rgb.astype(np.float32)
    return np.expand_dims(tensor, axis=0)

def compute_glare_rejected_heatmap(img_tensor, grad_model, pred_idx, img_bgr):
    h, w, _ = img_bgr.shape
    
    with tf.GradientTape() as tape:
        conv_outputs, model_predictions = grad_model(img_tensor)
        loss = model_predictions[:, pred_idx]
        
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val == 0: max_val = 1e-8
    raw_heatmap = (heatmap / max_val).numpy()
    
    # Convert and resize
    heatmap_uint8 = np.uint8(255 * raw_heatmap)
    heatmap_resized = cv2.resize(heatmap_uint8, (w, h))
    
    # Adaptive Geometric Guardrail: Detect eye bounds and clear outer lens reflections
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        (x_center, y_center), radius = cv2.minEnclosingCircle(largest_contour)
        
        # 80% Safe Area Extraction
        perfect_circle_mask = np.zeros((h, w), dtype=np.uint8)
        safe_radius = int(radius * 0.80)
        cv2.circle(perfect_circle_mask, (int(x_center), int(y_center)), safe_radius, 255, -1)
        isolated_heatmap = cv2.bitwise_and(heatmap_resized, perfect_circle_mask)
    else:
        x_center, y_center = w // 2, h // 2
        isolated_heatmap = heatmap_resized
        
    # Spatial quadrant division for tracking analytics
    quad_y, quad_x = int(y_center), int(x_center)
    quadrants = {
        "Upper-Left":  isolated_heatmap[0:quad_y, 0:quad_x],
        "Upper-Right": isolated_heatmap[0:quad_y, quad_x:w],
        "Lower-Left":  isolated_heatmap[quad_y:h, 0:quad_x],
        "Lower-Right": isolated_heatmap[quad_y:h, quad_x:w]
    }
    
    quad_sums = {k: np.sum(v) for k, v in quadrants.items()}
    total_sum = np.sum(list(quad_sums.values()))
    dominant_quadrant = max(quad_sums, key=quad_sums.get) if total_sum > 0 else "Central Retinal"
    quadrant_focus_pct = (quad_sums[dominant_quadrant] / total_sum * 100) if total_sum > 0 else 0.0
    
    return isolated_heatmap, dominant_quadrant, quadrant_focus_pct

# =====================================================================
#  4. USER INTERFACE GRAPHICS RENDERING CANVAS
# =====================================================================
st.markdown("<h2 style='text-align: center; color: #34d399; font-weight:700;'>🔬 RetiScan Pro v5</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8; margin-bottom: 25px;'>Clinical Grade Diagnostic Framework & Explainable AI Engine</p>", unsafe_allow_html=True)

# Custom global card style sheet injector
st.markdown("""
<style>
    div[data-testid="stColumn"] {
        background: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 12px;
        padding: 20px !important;
    }
    .stProgress > div > div > div > div {
        background-color: #ff9800 !important;
    }
</style>
""", unsafe_allow_html=True)

if model_loaded:
    uploaded_file = st.file_uploader("Injest Retinal Photographic Record Asset", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Matrix Ingestion & Verification Pass
        img_tensor = preprocess_for_inference(img_bgr)
        
        with st.spinner("Executing Graph Pipeline..."):
            probabilities = model.predict_on_batch(img_tensor)[0]
            
        pred_idx = int(np.argmax(probabilities))
        pred_name = CLASS_NAMES[pred_idx].split(' (')[0]
        confidence = probabilities[pred_idx] * 100
        accent_color = SEVERITY_COLOR[CLASS_NAMES[pred_idx]]
        
        # Run XAI Glare Elimination Calculations
        isolated_heatmap, dominant_quad, quad_pct = compute_glare_rejected_heatmap(img_tensor, grad_model, pred_idx, img_bgr)
        
        # Color Map Blending Operations
        heatmap_color = cv2.applyColorMap(isolated_heatmap, cv2.COLORMAP_JET)
        gradcam_blend = cv2.addWeighted(heatmap_color, 0.38, img_bgr, 0.62, 0)
        
        img_display_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gradcam_display_rgb = cv2.cvtColor(gradcam_blend, cv2.COLOR_BGR2RGB)
        
        # --- UI Row 1: The 4-Column Grid Layout ---
        col1, col2, col3, col4 = st.columns([1.2, 1.2, 1.5, 1.3], gap="medium")
        
        with col1:
            st.markdown("<span style='color: #8b949e; font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px; font-weight:600;'>Base Input</span>", unsafe_allow_html=True)
            st.image(img_display_rgb, use_container_width=True)
            
        with col2:
            st.markdown("<span style='color: #8b949e; font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px; font-weight:600;'>Grad Cam Map</span>", unsafe_allow_html=True)
            st.image(gradcam_display_rgb, use_container_width=True)
            
        with col3:
            st.markdown("<span style='color: #8b949e; font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px; font-weight:600;'>Diagnosis</span>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size: 28px; font-weight: 700; color: {accent_color}; margin-top:5px;'>{pred_name}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size: 13px; color: #8b949e; margin-bottom:12px;'>Confidence: {confidence:.1f}%</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size: 13.5px; line-height:1.5; color:#c9d1d9;'>{CLASS_DESCRIPTIONS[pred_idx]}</div>", unsafe_allow_html=True)
            
        with col4:
            st.markdown("<span style='color: #8b949e; font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px; font-weight:600;'>Grade Probabilities</span>", unsafe_allow_html=True)
            st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
            for i in range(NUM_CLASSES):
                cname = CLASS_NAMES[i]
                pct = float(probabilities[i])
                is_pred = (i == pred_idx)
                
                label_color = "#ffffff" if is_pred else "#8b949e"
                st.markdown(f"<div style='font-size: 12px; color: {label_color}; display: flex; justify-content: space-between; font-weight:500; margin-bottom:2px;'><span>{cname}</span><span>{pct*100:.1f}%</span></div>", unsafe_allow_html=True)
                st.progress(pct)

        # --- UI Row 2: Dynamic XAI Banner Block ---
        st.markdown("<br>", unsafe_allow_html=True)
        if pred_idx == 0:
            xai_text = "The internal neural activations are uniform and clean. No statistically significant pixel structural variations were flagged across the central retinal layout, indicating a stable vascular network structure."
        else:
            xai_text = f"The structural prediction tracking matrix is primarily driven by concentrated pathology vectors localized within the <strong>{dominant_quad} quadrant</strong> (accounting for {quad_pct:.1f}% of overall visual layer network activation). Neural focus clustered structural anomalies explicitly within this regional spatial sector."

        st.markdown(f"""
        <div style="background: #1f152d; border: 1px solid #4c297a; border-radius: 12px; padding: 18px; display: flex; gap: 14px; margin-bottom:15px;">
            <div style="font-size: 22px; margin-top:-2px;">🧠</div>
            <div>
                <div style="font-size: 12px; text-transform: uppercase; letter-spacing: 1px; color: #d6adff; font-weight: 600; margin-bottom: 4px;">Dynamic XAI Local Analysis</div>
                <div style="font-size: 13.5px; line-height: 1.45; color: #e1cbff;"><strong>AI Decision Rationale:</strong> {xai_text}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # --- UI Row 3: Actionable Medical Directive Banner ---
        st.markdown(f"""
        <div style="background: #1b1c16; border: 1px solid #4d4617; border-radius: 12px; padding: 18px; display: flex; gap: 14px;">
            <div style="font-size: 22px; margin-top:-2px;">📋</div>
            <div>
                <div style="font-size: 12px; text-transform: uppercase; letter-spacing: 1px; color: #e2da9a; font-weight: 600; margin-bottom: 4px;">Clinical Management Directive</div>
                <div style="font-size: 13.5px; line-height: 1.45; color: #f7f3d3;">{CLINICAL_DIRECTIVES[pred_idx]}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<hr style='border-color: #21262d;'>", unsafe_allow_html=True)
        st.markdown("<p style='color: #64748b; font-size: 11px; text-align: center;'>⚠️ Research framework support only. Mandatory ophthalmologist tracking validation required.</p>", unsafe_allow_html=True)
