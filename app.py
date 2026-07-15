import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import gdown
import json
import matplotlib.pyplot as plt
from datetime import datetime
from fpdf import FPDF
from skimage.filters import frangi # Mathematical vascular extraction

# Set page configurations to clean wide layout
st.set_page_config(page_title="RetiScan Pro v5", layout="wide", initial_sidebar_state="expanded")

# =====================================================================
#  1. CONSTANTS, CLOUD PATHS, & CORE METADATA
# =====================================================================
IMG_SIZE = 224
NUM_CLASSES = 5
MODEL_FILENAME = "retiscan_pro_v5_best.keras"
FILE_ID = "1NFcXDWOMIVyVbA9j2pXUR6b8kCYGVKyq"
HISTORY_FILE = "patient_history.json"

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
#  2. CACHED MODEL ENGINE & LONGITUDINAL DATABASE UTILITIES
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

def load_patient_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f: return json.load(f)
        except: return {}
    return {}

def save_patient_record(p_id, diagnosis, confidence, attention_index):
    history = load_patient_history()
    if p_id not in history: history[p_id] = []
    new_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "diagnosis": diagnosis,
        "confidence": round(float(confidence), 2),
        "attention_index": round(float(attention_index), 2)
    }
    history[p_id].append(new_entry)
    with open(HISTORY_FILE, "w") as f: json.dump(history, f, indent=4)
    return history[p_id]

# =====================================================================
#  3. PROCESSING ENGINE & METRICS (IQA, VASCULAR, TTA, PDF)
# =====================================================================
def preprocess_for_inference(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
    tensor = rgb.astype(np.float32)
    return np.expand_dims(tensor, axis=0)

def run_tta_ensemble_inference(model, base_tensor):
    pred_base = model.predict_on_batch(base_tensor)[0]
    flipped_tensor = np.flip(base_tensor, axis=2)
    pred_flipped = model.predict_on_batch(flipped_tensor)[0]
    gamma_tensor = np.clip(base_tensor * 1.05, 0.0, 255.0)
    pred_gamma = model.predict_on_batch(gamma_tensor)[0]

    fused_probabilities = (pred_base + pred_flipped + pred_gamma) / 3.0

    variance = np.var([pred_base, pred_flipped, pred_gamma], axis=0)
    mean_variance = np.mean(variance)
    consensus_badge = "HIGH CONSENSUS" if mean_variance < 0.02 else "BORDERLINE VERIFICATION REQUIRED"

    return fused_probabilities, consensus_badge

def generate_clinical_pdf(p_id, verdict, conf, attn_idx, quad, quad_pct, directive, consensus):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(16, 185, 129)
    pdf.cell(0, 10, "RETISCAN PRO DIAGNOSTIC SUMMARY REPORT", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, f"Patient Tracking Key: {p_id}", ln=True)
    pdf.cell(0, 8, f"Generated Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.cell(0, 8, f"Ensemble Integrity State: {consensus}", ln=True)
    pdf.line(10, pdf.get_y() + 2, 200, pdf.get_y() + 2)
    pdf.ln(8)

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, f"Diagnostic Conclusion: {verdict}", ln=True)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, f"Statistical Classification Confidence: {conf:.2f}%", ln=True)
    pdf.cell(0, 8, f"AI Neuro-Attention Mapping Index: {attn_idx:.1f}%", ln=True)
    pdf.cell(0, 8, f"Dominant Pathology Cluster: {quad} Quadrant ({quad_pct:.1f}% Focus)", ln=True)
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Official Management Protocol Directive:", ln=True)
    pdf.set_font("Helvetica", "I", 11)
    pdf.multi_cell(0, 6, directive)

    # Cast to bytes to ensure Streamlit download button compatibility
    return bytes(pdf.output())

def run_pre_computing_screening(img_bgr):
    h, w, _ = img_bgr.shape
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1. IQA Blur Gate
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < 45.0:
        return False, f"REJECTED: Asset fails focal clarity standards (Blur Variance: {blur_score:.1f}). Please recapture.", None, None, None

    # 2. Geometry & Luminance Checks
    mean_brightness = np.mean(gray)
    if mean_brightness < 10:
        return False, "REJECTED: Low asset luminance exposure (Image too dark).", None, None, None

    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return False, "REJECTED: Geometry unverified. No structural contour field found.", None, None, None

    largest_contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(largest_contour)
    total_area = h * w

    if contour_area < (total_area * 0.15):
        return False, "REJECTED: Geometry unverified. Extracted fundus structure area is too small.", None, None, None

    (x_center, y_center), radius = cv2.minEnclosingCircle(largest_contour)

    # --- NEW: 3. Biological Pigmentation Gate ---
    # Create a mask to only look at the inside of the circular object (ignoring the background)
    circle_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(circle_mask, (int(x_center), int(y_center)), int(radius * 0.8), 255, -1)

    # Calculate the average Blue, Green, and Red channel values inside that circle
    mean_b, mean_g, mean_r, _ = cv2.mean(img_bgr, mask=circle_mask)

    # Biological retinas are highly red-dominant with very low blue reflection.
    # If the blue channel is too high, or red isn't the dominant color, it's not a retina.
    if mean_b > 90 or mean_r < mean_g:
        return False, f"REJECTED: Biological profile mismatch. Asset lacks standard retinal pigmentation signatures (High anomalous blue/green reflection detected).", None, None, None

    return True, "PASSED", x_center, y_center, radius

def generate_vascular_map(img_bgr, x_center, y_center, radius):
    h, w, _ = img_bgr.shape
    _, g, _ = cv2.split(img_bgr)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_g = clahe.apply(g)
    inverted_g = cv2.bitwise_not(enhanced_g)

    vesselness = frangi(inverted_g, sigmas=range(1, 4, 1), black_ridges=False)
    vesselness_norm = cv2.normalize(vesselness, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    perfect_circle_mask = np.zeros((h, w), dtype=np.uint8)
    safe_radius = int(radius * 0.92)
    cv2.circle(perfect_circle_mask, (int(x_center), int(y_center)), safe_radius, 255, -1)

    clean_vessel_map = cv2.bitwise_and(vesselness_norm, perfect_circle_mask)
    return clean_vessel_map

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

    perfect_circle_mask = np.zeros((h, w), dtype=np.uint8)
    safe_radius = int(radius * 0.80)
    cv2.circle(perfect_circle_mask, (int(x_center), int(y_center)), safe_radius, 255, -1)
    isolated_heatmap = cv2.bitwise_and(heatmap_resized, perfect_circle_mask)

    if pred_idx == 0:
        ai_attention_index = 0.0
        boundary_img_bgr = img_bgr.copy()
    else:
        active_pixels = isolated_heatmap[isolated_heatmap > 0]
        if len(active_pixels) > 0:
            top_threshold = np.percentile(active_pixels, 90)
            ai_attention_index = np.mean(active_pixels[active_pixels >= top_threshold]) / 255.0 * 100.0
        else:
            ai_attention_index = 0.0

        max_internal_val = np.max(isolated_heatmap) if np.max(isolated_heatmap) > 0 else 1
        _, binary_mask = cv2.threshold(isolated_heatmap, int(0.55 * max_internal_val), 255, cv2.THRESH_BINARY)
        lesion_contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boundary_img_bgr = img_bgr.copy()
        mask_overlay = np.zeros_like(img_bgr)

        for contour in lesion_contours:
            area = cv2.contourArea(contour)
            if area > 25:
                cv2.drawContours(mask_overlay, [contour], -1, (255, 255, 0), -1)
                cv2.drawContours(boundary_img_bgr, [contour], -1, (255, 255, 0), 1)
                x, y, box_w, box_h = cv2.boundingRect(contour)
                cv2.rectangle(boundary_img_bgr, (x, y), (x + box_w, y + box_h), (0, 255, 255), 2)
                cv2.putText(boundary_img_bgr, "ROI FOCUS", (x, max(y - 8, 20)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        cv2.addWeighted(mask_overlay, 0.25, boundary_img_bgr, 0.75, 0, dst=boundary_img_bgr)

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

    return isolated_heatmap, boundary_img_bgr, ai_attention_index, dominant_quadrant, quadrant_focus_pct

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

st.sidebar.title("🩺 Clinical Control Hub")
st.sidebar.markdown("---")
patient_id = st.sidebar.text_input("👤 Patient ID Key", value="PATIENT-601").strip().upper()

if model_loaded:
    uploaded_file = st.sidebar.file_uploader("Upload Retinal Record Asset", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        passed_screening, message, x_center, y_center, radius = run_pre_computing_screening(img_bgr)

        if not passed_screening:
            st.markdown(f"""
            <div style='background:#7f1d1d; color:#fca5a5; padding:18px; border-radius:8px; font-weight:bold; border: 1px solid #f87171; margin-top:15px;'>
                ❌ SYSTEM SCREENING REJECTION: {message}<br>
                <span style='font-size:12px; font-weight:normal;'>Pipeline terminated automatically to prevent false model classification predictions on corrupted data configurations.</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            img_tensor = preprocess_for_inference(img_bgr)

            with st.spinner("Processing Multi-Variant Ensemble Imaging Analytics..."):
                probabilities, consensus_status = run_tta_ensemble_inference(model, img_tensor)

            pred_idx = int(np.argmax(probabilities))
            pred_name = CLASS_NAMES[pred_idx]
            confidence = probabilities[pred_idx] * 100
            accent_color = SEVERITY_COLOR[pred_name]

            isolated_heatmap, boundary_img, attention_index, dominant_quad, quad_pct = compute_diagnostic_graphs(
                img_tensor, grad_model, pred_idx, img_bgr, x_center, y_center, radius
            )

            with st.spinner("Mapping Vascular Topography..."):
                vessel_map = generate_vascular_map(img_bgr, x_center, y_center, radius)

            record_logs = save_patient_record(patient_id, pred_name, confidence, attention_index)

            heatmap_color = cv2.applyColorMap(isolated_heatmap, cv2.COLORMAP_JET)
            gradcam_blend = cv2.addWeighted(heatmap_color, 0.38, img_bgr, 0.62, 0)

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            gradcam_rgb = cv2.cvtColor(gradcam_blend, cv2.COLOR_BGR2RGB)
            boundary_rgb = cv2.cvtColor(boundary_img, cv2.COLOR_BGR2RGB)

            # === ROW 1: IMAGING HORIZONTAL PANEL ===
            st.markdown("<h4 style='color: #8b949e; font-size:13px; text-transform:uppercase; letter-spacing:1px; margin-bottom:10px;'>Multi-Panel Diagnostic Worksheet Trackers</h4>", unsafe_allow_html=True)
            img_col1, img_col2, img_col3, img_col4, img_col5 = st.columns(5, gap="medium")
            # === ROW 1: PRIMARY DIAGNOSTIC VISUALS ===
            st.markdown("<h4 style='color: #8b949e; font-size:13px; text-transform:uppercase; letter-spacing:1px; margin-bottom:10px;'>Primary Diagnostic Trackers</h4>", unsafe_allow_html=True)
            
            # Reduce to 3 main columns for a cleaner, wider layout
            img_col1, img_col2, img_col3 = st.columns(3, gap="large")

            with img_col1:
                st.markdown("<span style='color: #8b949e; font-size: 11px; text-transform: uppercase; font-weight:600;'>I. Base Input</span>", unsafe_allow_html=True)
                st.image(img_rgb, use_container_width=True)

            with img_col2:
                st.markdown("<span style='color: #8b949e; font-size: 11px; text-transform: uppercase; font-weight:600;'>II. Vascular Topology Map</span>", unsafe_allow_html=True)
                st.image(vessel_map, use_container_width=True, clamp=True)
                
            with img_col3:
                st.markdown("<span style='color: #8b949e; font-size: 11px; text-transform: uppercase; font-weight:600;'>III. Grad-CAM Path Map</span>", unsafe_allow_html=True)
                st.markdown("<span style='color: #8b949e; font-size: 11px; text-transform: uppercase; font-weight:600;'>II. Grad-CAM Path Map</span>", unsafe_allow_html=True)
                st.image(gradcam_rgb, use_container_width=True)

            with img_col4:
                st.markdown(f"<span style='color: #00FFFF; font-size: 11px; text-transform: uppercase; font-weight:600;'>IV. AI Focus ({attention_index:.1f} Index)</span>", unsafe_allow_html=True)
                st.image(boundary_rgb, use_container_width=True)
                
            with img_col5:
                st.markdown("<span style='color: #38bdf8; font-size: 11px; text-transform: uppercase; font-weight:600;'>V. Diagnostics Forecast</span>", unsafe_allow_html=True)
            with img_col3:
                st.markdown("<span style='color: #38bdf8; font-size: 11px; text-transform: uppercase; font-weight:600;'>III. Diagnostics Forecast</span>", unsafe_allow_html=True)
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

                    trajectory_alert = "ACCELERATING" if slope > 1.5 else "STABILIZED"
                else:
                    trajectory_alert = "INSUFFICIENT_TIMELINE_DATA"

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

            # === ON-DEMAND EXPANDER FOR SECONDARY VISUALS ===
            with st.expander("🔍 View Advanced Technical Layers (Vascular & AI ROI Focus)"):
                adv_col1, adv_col2 = st.columns(2, gap="medium")
                with adv_col1:
                    st.markdown("<span style='color: #8b949e; font-size: 11px; text-transform: uppercase; font-weight:600;'>Vascular Topology Map</span>", unsafe_allow_html=True)
                    st.image(vessel_map, use_container_width=True, clamp=True)
                with adv_col2:
                    st.markdown(f"<span style='color: #00FFFF; font-size: 11px; text-transform: uppercase; font-weight:600;'>AI ROI Bounding Boxes ({attention_index:.1f} Index)</span>", unsafe_allow_html=True)
                    st.image(boundary_rgb, use_container_width=True)

            # === ROW 2: DIAGNOSTIC MATRIX SUMMARY ===
            st.markdown("<br><h4 style='color: #8b949e; font-size:13px; text-transform:uppercase; letter-spacing:1px; margin-bottom:10px;'>Classification & Metrics Summary</h4>", unsafe_allow_html=True)
            data_col1, data_col2 = st.columns([1, 1], gap="medium")

            with data_col1:
                st.markdown("<span style='color: #8b949e; font-size: 11px; text-transform: uppercase; font-weight:600;'>Diagnostic Verdict</span>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size: 32px; font-weight: 700; color: {accent_color}; margin-top:5px;'>{pred_name}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size: 12px; color: #8b949e; margin-bottom:12px;'>Integrity Consensus Validation: <b style='color:#38bdf8;'>{consensus_status}</b></div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size: 14px; color: #8b949e; margin-bottom:12px;'>Confidence: {confidence:.2f}% | Attention Intensity Score: {attention_index:.1f}%</div>", unsafe_allow_html=True)
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

            # === ROW 3: EXPLAINABLE AI BANNERS WITH PREDICTIVE TRACKING ===
            st.markdown("<br>", unsafe_allow_html=True)
            if pred_idx == 0:
                xai_text = "The internal neural activations are uniform and clean. No statistically significant anomalous pixel groupings were identified, indicating a stable vascular layer context."
            else:
                xai_text = f"The structural prediction tracking matrix is primarily driven by concentrated pathology vectors localized within the <strong>{dominant_quad} quadrant</strong> (accounting for {quad_pct:.1f}% of overall visual network attention maps)."

                if len(record_logs) >= 2:
                    if trajectory_alert == "ACCELERATING":
                        xai_text += f" <span style='color:#ef4444; font-weight:bold;'>WARNING: Longitudinal tracking indicates an upward pathology velocity (+{slope:.1f}% index shift per visit). Interventions should be accelerated immediately.</span>"
                    else:
                        xai_text += " Longitudinal tracking denotes a stabilized tracking vector path distribution."

            st.markdown(f"""
            <div style="background: #1f152d; border: 1px solid #4c297a; border-radius: 12px; padding: 18px; display: flex; gap: 14px; margin-bottom:15px;">
                <div style="font-size: 22px; margin-top:-2px;">🧠</div>
                <div>
                    <div style="font-size: 12px; text-transform: uppercase; letter-spacing: 1px; color: #d6adff; font-weight: 600; margin-bottom: 4px;">Dynamic XAI & Predictive Prognostics</div>
                    <div style="font-size: 13.5px; line-height: 1.45; color: #e1cbff;"><strong>AI Decision Rationale:</strong> {xai_text}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background: #1b1c16; border: 1px solid #4d4617; border-radius: 12px; padding: 18px; display: flex; gap: 14px; margin-bottom:25px;">
                <div style="font-size: 22px; margin-top:-2px;">📋</div>
                <div>
                    <div style="font-size: 12px; text-transform: uppercase; letter-spacing: 1px; color: #e2da9a; font-weight: 600; margin-bottom: 4px;">Clinical Management Directive</div>
                    <div style="font-size: 13.5px; line-height: 1.45; color: #f7f3d3;">{CLINICAL_DIRECTIVES[pred_idx]}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # === ROW 4: DATA PAYLOAD LOGGING TABLE ===
            st.subheader(f"📋 Historical Tracking Summary Log ({patient_id})")

            table_html = '<table style="width:100%; text-align:left; border-collapse:collapse; font-size:14px; background:#161b22; border:1px solid #30363d; border-radius:8px;">'
            table_html += '<thead><tr style="border-bottom:2px solid #30363d; color:#8b949e; background:#0d1117;">'
            table_html += '<th style="padding:12px;">Visit Index</th>'
            table_html += '<th style="padding:12px;">Timestamp</th>'
            table_html += '<th style="padding:12px;">Diagnosis Verdict</th>'
            table_html += '<th style="padding:12px;">Confidence Score</th>'
            table_html += '<th style="padding:12px;">Attention Index</th>'
            table_html += '</tr></thead><tbody>'

            for i, r in enumerate(record_logs):
                table_html += '<tr style="border-bottom:1px solid #30363d;">'
                table_html += f'<td style="padding:12px;"><b>#{i+1}</b></td>'
                table_html += f'<td style="padding:12px;">{r["timestamp"]}</td>'
                table_html += f'<td style="padding:12px;">{r["diagnosis"]}</td>'
                table_html += f'<td style="padding:12px;">{r["confidence"]}%</td>'
                table_html += f'<td style="padding:12px; color:#38bdf8;"><b>{r["attention_index"]:.1f}%</b></td>'
                table_html += '</tr>'

            table_html += '</tbody></table>'
            st.markdown(table_html, unsafe_allow_html=True)

            # --- DYNAMIC EXPORT BUTTON ASSEMBLIES ---
            st.markdown("<br>", unsafe_allow_html=True)
            pdf_bytes = generate_clinical_pdf(
                patient_id, pred_name, confidence, attention_index, 
                dominant_quad, quad_pct, CLINICAL_DIRECTIVES[pred_idx], consensus_status
            )

            st.download_button(
                label="📥 Export Signed Clinical Summary PDF",
                data=pdf_bytes,
                file_name=f"RetiScan_{patient_id}_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
else:
    st.info("👋 Welcome! Please upload a retinal fundus photo inside the left sidebar panel to initialize the tracking dashboard timeline sequence workflows.")
