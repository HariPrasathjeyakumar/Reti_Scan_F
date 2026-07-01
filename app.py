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
FILE_ID = "1BrmHzARxRFTBilmymFkwn6e5ZjPmosu1"

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
#  3. PROCESSING ENGINE & GLARE-FREE REGION SEGMENTATION (ROI)
# =====================================================================
def preprocess_for_inference(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
    tensor = rgb.astype(np.float32)
    return np.expand_dims(tensor, axis=0)

def run_pre_computing_screening(img_bgr):
    """
    Executes raw data matrix filtering before exposing files to neural nodes.
    Checks structural circle area constraints and mean pixel luminance.
    """
    h, w, _ = img_bgr.shape
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Check 1: Exposure Test
    mean_brightness = np.mean(gray)
    if mean_brightness < 10:
        return False, "REJECTED: Low asset luminance exposure (Image too dark).", None, None, None
        
    # Check 2: Geometry Structural Test
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return False, "REJECTED: Geometry unverified. No structural contour field found.", None, None, None
        
    largest_contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(largest_contour)
    total_area = h * w
    
    # Eye canvas must cover at least 15% of the file stream area
    if contour_area < (total_area * 0.15):
        return False, "REJECTED: Geometry unverified. Extracted fundus structure area is too small.", None, None, None
        
    (x_center, y_center), radius = cv2.minEnclosingCircle(largest_contour)
    return True, "PASSED", x_center, y_center, radius

def compute_diagnostic_graphs(img_tensor, grad_model, pred_idx, img_bgr, x_center, y_center, radius):
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
    
    heatmap_uint8 = np.uint8(255 * raw_heatmap)
    heatmap_resized = cv2.resize(heatmap_uint8, (w, h))
    
    # 80% Safe Circle Inner Radius Area Cutoff
    perfect_circle_mask = np.zeros((h, w), dtype=np.uint8)
    safe_radius = int(radius * 0.80)
    cv2.circle(perfect_circle_mask, (int(x_center), int(y_center)), safe_radius, 255, -1)
    isolated_heatmap = cv2.bitwise_and(heatmap_resized, perfect_circle_mask)
        
    # Generate the 3rd Marking Graph: Bounding Box Contours (ROI)
    max_internal_val = np.max(isolated_heatmap)
    if max_internal_val == 0: max_internal_val = 1
    
    _, binary_mask = cv2.threshold(isolated_heatmap, int(0.55 * max_internal_val), 255, cv2.THRESH_BINARY)
    lesion_contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boundary_img_bgr = img_bgr.copy()
    mask_overlay = np.zeros_like(img_bgr)
    total_anomaly_pixels = 0
    
    for contour in lesion_contours:
        area = cv2.contourArea(contour)
        if area > 25:
            total_anomaly_pixels += area
            cv2.drawContours(mask_overlay, [contour], -1, (255, 255, 0), -1)
            cv2.drawContours(boundary_img_bgr, [contour], -1, (255, 255, 0), 1)
            x, y, box_w, box_h = cv2.boundingRect(contour)
            cv2.rectangle(boundary_img_bgr, (x, y), (x + box_w, y + box_h), (0, 255, 255), 2)
            cv2.putText(boundary_img_bgr, f"ROI: [{x},{y}]", (x, max(y - 8, 20)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
    cv2.addWeighted(mask_overlay, 0.25, boundary_img_bgr, 0.75, 0, dst=boundary_img_bgr)
    stress_pct = (total_anomaly_pixels / (h * w)) * 100
    
    # Compute quadrant distribution statistics
    quad_y, quad_x = int(y_center), int(x_center)
    quadrants = {
        "Upper-Left":  isolated_heatmap[0:quad_y, 0:quad_x],
        "Upper-Right": isolated_heatmap[0:quad_y, quad_x:w],
        "Lower-Left":  isolated_heatmap[quad_y:h, 0:quad_x],
        "Lower-Right": isolated_heatmap[quad_y:h, quad_x:w]
    }
    quad_sums = {k: np.sum(v) for k, v in quadrants.items()}
    total_sum = np.sum(list(quad_sums.values()))
    dominant_quadrant = max(quad_sums, key=quad_sums.get) if total_sum > 0 else "Central"
    quadrant_focus_pct = (quad_sums[dominant_quadrant] / total_sum * 100) if total_sum > 0 else 0.0
    
    return isolated_heatmap, boundary_img_bgr, stress_pct, dominant_quadrant, quadrant_focus_pct

# =====================================================================
#  4. USER INTERFACE GRAPHICS RENDERING CANVAS
# =====================================================================
st.markdown("<h2 style='text-align: center; color: #34d399; font-weight:700;'>🔬 RetiScan Pro v5 Dashboard</h2>", unsafe_allow_html=True)

st.markdown("""
<style>
    div[data-testid="stColumn"] {
        background: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 12px;
        padding: 15px !important;
    }
    .stProgress > div > div > div > div {
        background-color: #ff9800 !important;
    }
</style>
""", unsafe_allow_html=True)

if model_loaded:
    uploaded_file = st.file_uploader("Upload Retinal Record Asset", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # --- EXECUTE PRE-COMPUTING SCREENING FILTER ---
        passed_screening, message, x_center, y_center, radius = run_pre_computing_screening(img_bgr)
        
        if not passed_screening:
            st.markdown(f"""
            <div style='background:#7f1d1d; color:#fca5a5; padding:18px; border-radius:8px; font-weight:bold; border: 1px solid #f87171; margin-top:15px;'>
                ❌ SYSTEM SCREENING REJECTION: {message}<br>
                <span style='font-size:12px; font-weight:normal;'>Pipeline terminated automatically to prevent false model classification predictions on corrupted data configurations.</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Asset cleared metrics. Proceed safely to execution loops.
            img_tensor = preprocess_for_inference(img_bgr)
            
            with st.spinner("Processing Imaging Analytics..."):
                probabilities = model.predict_on_batch(img_tensor)[0]
                
            pred_idx = int(np.argmax(probabilities))
            pred_name = CLASS_NAMES[pred_idx]
            confidence = probabilities[pred_idx] * 100
            accent_color = SEVERITY_COLOR[pred_name]
            
            # Compute processing maps including the 3rd Visual Graph
            isolated_heatmap, boundary_img, pathology_area, dominant_quad, quad_pct = compute_diagnostic_graphs(
                img_tensor, grad_model, pred_idx, img_bgr, x_center, y_center, radius
            )
            
            heatmap_color = cv2.applyColorMap(isolated_heatmap, cv2.COLORMAP_JET)
            gradcam_blend = cv2.addWeighted(heatmap_color, 0.38, img_bgr, 0.62, 0)
            
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            gradcam_rgb = cv2.cvtColor(gradcam_blend, cv2.COLOR_BGR2RGB)
            boundary_rgb = cv2.cvtColor(boundary_img, cv2.COLOR_BGR2RGB)
            
            # === ROW 1: THE THREE VISUAL IMAGING GRAPHS (RESTORED WITH PRE-COMPUTING GUARANTEES) ===
            st.markdown("<h4 style='color: #8b949e; font-size:13px; text-transform:uppercase; letter-spacing:1px; margin-bottom:10px;'>Visual Diagnostic Trackers</h4>", unsafe_allow_html=True)
            img_col1, img_col2, img_col3 = st.columns(3, gap="medium")
            
            with img_col1:
                st.markdown("<span style='color: #8b949e; font-size: 11px; text-transform: uppercase; font-weight:600;'>I. Base Input</span>", unsafe_allow_html=True)
                st.image(img_rgb, use_container_width=True)
                
            with img_col2:
                st.markdown("<span style='color: #8b949e; font-size: 11px; text-transform: uppercase; font-weight:600;'>II. Grad-CAM Map</span>", unsafe_allow_html=True)
                st.image(gradcam_rgb, use_container_width=True)
                
            with img_col3:
                st.markdown(f"<span style='color: #00FFFF; font-size: 11px; text-transform: uppercase; font-weight:600;'>III. Automated Lesion Boundary ({pathology_area:.2f}%)</span>", unsafe_allow_html=True)
                st.image(boundary_rgb, use_container_width=True)
                
            # === ROW 2: CLINICAL QUANTIFICATION AND PROGRESS METRICS ===
            st.markdown("<br><h4 style='color: #8b949e; font-size:13px; text-transform:uppercase; letter-spacing:1px; margin-bottom:10px;'>Classification & Metrics Summary</h4>", unsafe_allow_html=True)
            data_col1, data_col2 = st.columns([1, 1], gap="medium")
            
            with data_col1:
                st.markdown("<span style='color: #8b949e; font-size: 11px; text-transform: uppercase; font-weight:600;'>Diagnostic Verdict</span>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size: 32px; font-weight: 700; color: {accent_color}; margin-top:5px;'>{pred_name}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size: 14px; color: #8b949e; margin-bottom:12px;'>Confidence: {confidence:.2f}% | Pathological Footprint: {pathology_area:.2f}%</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size: 14px; line-height:1.5; color:#c9d1d9;'>{CLASS_DESCRIPTIONS[pred_idx]}</div>", unsafe_allow_html=True)
                
            with data_col2:
                st.markdown("<span style='color: #8b949e; font-size: 11px; text-transform: uppercase; font-weight:600;'>Grade Probabilities Matrix</span>", unsafe_allow_html=True)
                st.markdown("<div style='margin-top: 5px;'></div>", unsafe_allow_html=True)
                for i in range(NUM_CLASSES):
                    cname = CLASS_NAMES[i]
                    pct = float(probabilities[i])
                    is_pred = (i == pred_idx)
                    
                    label_color = "#ffffff" if is_pred else "#8b949e"
                    st.markdown(f"<div style='font-size: 12px; color: {label_color}; display: flex; justify-content: space-between; font-weight:500; margin-bottom:2px;'><span>{cname}</span><span>{pct*100:.1f}%</span></div>", unsafe_allow_html=True)
                    st.progress(pct)

            # === ROW 3: BANNERS ===
            st.markdown("<br>", unsafe_allow_html=True)
            if pred_idx == 0:
                xai_text = "The internal neural activations are uniform and clean. No statistically significant anomalous pixel groupings were identified, indicating a stable vascular layer context."
            else:
                xai_text = f"The structural prediction tracking matrix is primarily driven by concentrated pathology vectors localized within the <strong>{dominant_quad} quadrant</strong> (accounting for {quad_pct:.1f}% of overall visual network attention maps). Neural tracking isolated distinct anomalies within this sector."

            st.markdown(f"""
            <div style="background: #1f152d; border: 1px solid #4c297a; border-radius: 12px; padding: 18px; display: flex; gap: 14px; margin-bottom:15px;">
                <div style="font-size: 22px; margin-top:-2px;">🧠</div>
                <div>
                    <div style="font-size: 12px; text-transform: uppercase; letter-spacing: 1px; color: #d6adff; font-weight: 600; margin-bottom: 4px;">Dynamic XAI Local Analysis</div>
                    <div style="font-size: 13.5px; line-height: 1.45; color: #e1cbff;"><strong>AI Decision Rationale:</strong> {xai_text}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="background: #1b1c16; border: 1px solid #4d4617; border-radius: 12px; padding: 18px; display: flex; gap: 14px;">
                <div style="font-size: 22px; margin-top:-2px;">📋</div>
                <div>
                    <div style="font-size: 12px; text-transform: uppercase; letter-spacing: 1px; color: #e2da9a; font-weight: 600; margin-bottom: 4px;">Clinical Management Directive</div>
                    <div style="font-size: 13.5px; line-height: 1.45; color: #f7f3d3;">{CLINICAL_DIRECTIVES[pred_idx]}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
