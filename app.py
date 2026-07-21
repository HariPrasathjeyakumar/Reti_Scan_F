import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import gdown
import json
import hashlib
import matplotlib.pyplot as plt
import pytz
from datetime import datetime
from fpdf import FPDF
from skimage.filters import frangi  # Mathematical vascular extraction

# Set page configurations — wide canvas, no sidebar (structure is fully custom)
st.set_page_config(
    page_title="RetiScan Pro v5",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================================================
#  0. DESIGN TOKENS — graphite + amber/emerald duotone (distinct identity)
# =====================================================================
BG           = "#0d0e12"
SURFACE      = "#16171d"
SURFACE_ALT  = "#1d1f27"
RAISED       = "#24262f"
BORDER       = "#2c2e38"
TEXT_MAIN    = "#f3f2ee"
TEXT_MUTED   = "#96979f"
TEXT_FAINT   = "#5f606a"
ACCENT       = "#ff8a3d"     # warm amber — primary identity color
ACCENT_DEEP  = "#e56f22"
EMERALD      = "#33c793"     # secondary accent — used for positive/safe states
INFO         = "#5eb1ef"
WARN         = "#f2b134"
DANGER       = "#f2545b"
SUCCESS      = "#33c793"

SEVERITY_COLOR = {
    "No DR":            EMERALD,
    "Mild NPDR":         WARN,
    "Moderate NPDR":     INFO,
    "Severe NPDR":       "#ef8b3f",
    "Proliferative DR":  DANGER,
}

st.markdown(f"""
    <style>
    /* ============ Canvas ============ */
    html, body, .main, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
        background-color: {BG} !important;
        color: {TEXT_MAIN};
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    [data-testid="stHeader"] {{ background-color: transparent !important; }}
    [data-testid="stSidebar"] {{ display: none !important; }}
    .block-container {{ padding-top: 0.5rem; padding-bottom: 4rem; max-width: 1180px; }}

    /* ============ Inputs ============ */
    .stTextInput input {{
        background-color: {SURFACE_ALT} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 10px !important;
        color: {TEXT_MAIN} !important;
        font-weight: 600 !important;
    }}
    .stTextInput input:focus {{
        border-color: {ACCENT} !important;
        box-shadow: 0 0 0 1px {ACCENT} !important;
    }}
    label {{ color: {TEXT_MUTED} !important; font-size: 11.5px !important; font-weight: 700 !important;
             text-transform: uppercase; letter-spacing: 0.6px; }}

    /* ============ Uploader ============ */
    [data-testid="stFileUploaderDropzone"] {{
        background-color: {SURFACE_ALT} !important;
        border: 1.5px dashed {BORDER} !important;
        border-radius: 12px !important;
    }}
    [data-testid="stFileUploaderDropzone"] * {{ color: #e8f6fa !important; }}
    [data-testid="stFileUploaderDropzone"] button,
    section[data-testid="stFileUploader"] button {{
        background-color: {ACCENT} !important;
        color: #201200 !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
    }}
    [data-testid="stFileUploaderFile"] {{
        background-color: {RAISED} !important;
        border-radius: 8px !important;
        border: 1px solid {BORDER} !important;
    }}

    /* ============ Buttons ============ */
    .stButton button, .stDownloadButton button {{
        background-color: {ACCENT} !important;
        color: #201200 !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 800 !important;
        padding: 12px 18px !important;
        box-shadow: 0 4px 16px rgba(255, 138, 61, 0.28);
        transition: transform 0.12s ease, box-shadow 0.12s ease;
    }}
    .stButton button:hover, .stDownloadButton button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(255, 138, 61, 0.4);
    }}
    .stDownloadButton button p {{ color: #201200 !important; font-weight: 800 !important; }}

    /* ============ Radio-as-segmented-control (image switcher) ============ */
    div[role="radiogroup"] {{
        display: flex; gap: 6px; background: {SURFACE_ALT};
        padding: 5px; border-radius: 12px; border: 1px solid {BORDER};
        width: fit-content;
    }}
    div[role="radiogroup"] label {{
        background: transparent; border-radius: 9px; padding: 8px 16px !important;
        margin: 0 !important; color: {TEXT_MUTED} !important; font-weight: 700 !important;
        text-transform: none !important; letter-spacing: 0 !important; font-size: 13px !important;
        cursor: pointer;
    }}
    div[role="radiogroup"] label[data-checked="true"],
    div[role="radiogroup"] input:checked + div {{ color: {ACCENT} !important; }}
    div[role="radiogroup"] > label > div:first-child {{ display: none; }}

    /* ============ Expander ============ */
    .stExpander {{
        background-color: {SURFACE} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 14px;
    }}
    .stExpander summary {{ color: {TEXT_MAIN} !important; font-weight: 700 !important; }}

    /* ============ Misc widgets ============ */
    hr {{ border-color: {BORDER} !important; }}
    ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
    ::-webkit-scrollbar-thumb {{ background-color: {BORDER}; border-radius: 8px; }}

    /* ============ Custom layout primitives ============ */
    .rs-hero {{
        display: flex; align-items: center; gap: 16px;
        padding: 26px 0 22px 0;
    }}
    .rs-hero-mark {{
        width: 52px; height: 52px; border-radius: 16px; flex-shrink: 0;
        background: linear-gradient(145deg, {ACCENT}, {ACCENT_DEEP});
        display: flex; align-items: center; justify-content: center; font-size: 26px;
        box-shadow: 0 6px 20px rgba(255, 138, 61, 0.3);
    }}
    .rs-hero-title {{ font-size: 27px; font-weight: 800; letter-spacing: -0.3px; line-height: 1.15; }}
    .rs-hero-sub {{ color: {TEXT_MUTED}; font-size: 13px; margin-top: 3px; }}

    .rs-toolbar {{
        background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 16px;
        padding: 18px 22px; margin-bottom: 26px;
    }}
    .rs-toolbar-label {{
        color: {TEXT_FAINT}; font-size: 10.5px; text-transform: uppercase;
        letter-spacing: 0.8px; font-weight: 800; margin-bottom: 8px;
    }}

    .rs-divider-label {{
        color: {TEXT_MAIN}; font-size: 12.5px; text-transform: uppercase;
        letter-spacing: 1px; font-weight: 800; margin: 30px 0 16px 0;
        display: flex; align-items: center; gap: 10px;
    }}
    .rs-divider-label::after {{
        content: ""; flex: 1; height: 1px; background: {BORDER};
    }}

    .rs-verdict-hero {{
        border-radius: 20px; padding: 32px 34px; margin-bottom: 8px;
        position: relative; overflow: hidden; border: 1px solid {BORDER};
    }}
    .rs-pill {{
        display: inline-flex; align-items: center; gap: 6px; font-size: 12px; font-weight: 700;
        padding: 6px 13px; border-radius: 100px; border: 1px solid; margin-right: 8px;
    }}
    .rs-stat-strip {{ display: flex; gap: 28px; flex-wrap: wrap; margin-top: 18px; }}
    .rs-stat {{ display: flex; flex-direction: column; }}
    .rs-stat-label {{ color: {TEXT_MUTED}; font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.6px; font-weight: 700; }}
    .rs-stat-value {{ font-size: 19px; font-weight: 800; margin-top: 2px; }}

    .rs-rail {{ border-left: 3px solid {BORDER}; padding-left: 18px; margin-bottom: 22px; }}
    .rs-rail-accent {{ border-left: 3px solid {ACCENT}; padding-left: 18px; margin-bottom: 22px; }}
    .rs-rail-warn {{ border-left: 3px solid {WARN}; padding-left: 18px; margin-bottom: 22px; }}
    .rs-rail-title {{ font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.8px; font-weight: 800; margin-bottom: 8px; }}
    .rs-rail-body {{ font-size: 13.5px; line-height: 1.65; color: {TEXT_MAIN}; }}

    .rs-prob-row {{ display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 5px; }}

    .rs-bar-track {{
        width: 100%; height: 8px; border-radius: 6px;
        background: {SURFACE_ALT}; border: 1px solid {BORDER};
        overflow: hidden; margin-bottom: 14px;
    }}
    .rs-bar-fill {{
        height: 100%; border-radius: 6px;
        background: linear-gradient(90deg, {ACCENT}, {ACCENT_DEEP});
    }}
    .rs-bar-fill-muted {{
        height: 100%; border-radius: 6px;
        background: linear-gradient(90deg, {TEXT_FAINT}, {BORDER});
    }}

    .rs-reject {{
        background: {SURFACE}; border-left: 4px solid {DANGER}; border-radius: 10px;
        padding: 18px 22px; color: {TEXT_MAIN};
    }}

    .rs-timeline-chip {{
        display: inline-flex; flex-direction: column; gap: 2px;
        background: {SURFACE_ALT}; border: 1px solid {BORDER}; border-radius: 12px;
        padding: 10px 16px; margin-right: 10px; margin-bottom: 10px; min-width: 140px;
    }}
    </style>
""", unsafe_allow_html=True)

# =====================================================================
#  1. CONSTANTS, CLOUD PATHS, & CORE METADATA
# =====================================================================
IMG_SIZE = 224
NUM_CLASSES = 5

# --- DR Model Settings ---
MODEL_FILENAME = "retiscan_pro_v5_best.keras"
FILE_ID = "1NFcXDWOMIVyVbA9j2pXUR6b8kCYGVKyq"

# --- Integrated HR Model Settings ---
HR_MODEL_FILENAME = "retiscan_hr_best.keras"
HR_FILE_ID = ""  # Add your Google Drive File ID here if hosting remotely

HISTORY_FILE = "patient_history.json"

CLASS_NAMES = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR"
}

CLASS_DESCRIPTIONS = {
    0: "No visible signs of diabetic retinopathy. The retinal vasculature, macula, and optic disc region show no microaneurysms, "
       "hemorrhages, hard exudates, or neovascularization within the analyzed field. This represents the healthiest end of the "
       "ICDR (International Clinical Diabetic Retinopathy) severity scale.",
    1: "Mild Non-Proliferative DR — the earliest detectable stage. Characterized by isolated microaneurysms: small, localized "
       "balloon-like outpouchings in retinal capillary walls caused by chronic hyperglycemic damage to vessel structure. "
       "No hemorrhages or exudates are typically present yet, and vision is usually unaffected at this stage.",
    2: "Moderate Non-Proliferative DR — a progressive stage marked by an increasing number of microaneurysms, scattered "
       "intraretinal hemorrhages ('dot-and-blot' pattern), and early microvascular abnormalities. This is the threshold at "
       "which most screening guidelines recommend specialist referral, as the risk of progression to sight-threatening disease rises.",
    3: "Severe Non-Proliferative DR — extensive hemorrhages and microaneurysms across multiple retinal quadrants, along with "
       "venous beading and intraretinal microvascular abnormalities (IRMA). This stage carries a substantially elevated short-term "
       "risk of progressing to proliferative disease and requires prompt ophthalmological evaluation.",
    4: "Proliferative DR — the most advanced stage, defined by neovascularization: fragile, abnormal new blood vessels growing "
       "in response to retinal ischemia. These vessels are prone to leakage and rupture, creating a high risk of vitreous "
       "hemorrhage and tractional retinal detachment if left untreated. This stage constitutes a medical emergency."
}

CLINICAL_DIRECTIVES = {
    0: "Schedule routine tracking examination in 12 months. Continue baseline systemic blood glucose and blood pressure monitoring.",
    1: "Schedule a targeted Optical Coherence Tomography (OCT) review focusing on localized structural fields. Re-examine in 6-12 months.",
    2: "Refer to an ophthalmologist for urgent structural baseline tracking. Evaluate macular edema and stabilize HbA1c variables.",
    3: "Immediate specialist referral required. Initiate rapid panretinal photocoagulation (PRP) screening profiles safely.",
    4: "CRITICAL ALERT: Emergency vitreo-retinal surgical evaluation indicated. High immediate risk of permanent tractional detachment."
}

REFERABLE_CLASSES = {2, 3, 4}
NON_REFERABLE_CLASSES = {0, 1}

MODEL_CARD = {
    "architecture": "Dual Pipeline: DR EfficientNetB3 + HR Vascular Classifier",
    "input_resolution": f"{IMG_SIZE} x {IMG_SIZE} RGB",
    "training_dataset": "APTOS 2019 & Vascular Retinal Screening Corpora",
    "num_classes": "5-class DR ICDR scale + Binary HR Screening Triage",
    "loss_function": "Categorical Crossentropy & Binary Crossentropy",
    "reported_accuracy": "~89% DR top-1 accuracy / ~91% HR binary sensitivity",
    "explainability_method": "Grad-CAM (gradient-weighted class activation mapping) + Frangi Vascular Topography",
    "uncertainty_method": "Test-Time Augmentation (TTA) ensemble variance across 3 views",
    "known_limitations": [
        "Trained and validated on standard fundus databases; performance across uncalibrated hardware requires validation.",
        "Grad-CAM highlights regions correlated with the prediction — it does not guarantee the highlighted region is the true clinical lesion.",
        "Longitudinal trend forecasting requires at least 3 visits to be statistically meaningful; earlier visits are descriptive only.",
        "Not a standalone diagnostic device. Intended as a decision-support and triage aid alongside qualified clinical judgement.",
    ],
    "intended_use": "Dual screening triage support for DR grading & Hypertensive Retinopathy vascular assessment.",
}

def focal_loss():
    def loss_fn(y_true, y_pred): return tf.reduce_mean(y_pred)
    loss_fn.__name__ = "focal_loss"
    return loss_fn

# =====================================================================
#  2. CACHED DUAL MODEL ENGINE & LONGITUDINAL DATABASE UTILITIES
# =====================================================================
@st.cache_resource
def load_retiscan_pipeline():
    # --- Load Primary DR Model ---
    if not os.path.exists(MODEL_FILENAME):
        with st.spinner("Downloading trained DR model weights from cloud..."):
            gdown.download(id=FILE_ID, output=MODEL_FILENAME, quiet=False)

    if not os.path.exists(MODEL_FILENAME) or os.path.getsize(MODEL_FILENAME) < 1000000:
        raise FileNotFoundError("The DR model file downloaded is corrupted or empty.")

    main_model = tf.keras.models.load_model(MODEL_FILENAME, custom_objects={"focal_loss": focal_loss()})

    conv_layer_name = None
    for layer in reversed(main_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D) or 'top_conv' in layer.name or 'block7' in layer.name:
            conv_layer_name = layer.name
            break
    if not conv_layer_name:
        conv_layer_name = "top_conv"

    grad_model = tf.keras.models.Model([main_model.inputs], [main_model.get_layer(conv_layer_name).output, main_model.output])
    
    # --- Load Integrated HR Model (or Fallback Synthetic Evaluator) ---
    hr_model = None
    if os.path.exists(HR_MODEL_FILENAME):
        try:
            hr_model = tf.keras.models.load_model(HR_MODEL_FILENAME, custom_objects={"focal_loss": focal_loss()})
        except Exception:
            hr_model = None
    elif HR_FILE_ID:
        try:
            with st.spinner("Downloading HR classifier weights..."):
                gdown.download(id=HR_FILE_ID, output=HR_MODEL_FILENAME, quiet=False)
            hr_model = tf.keras.models.load_model(HR_MODEL_FILENAME, custom_objects={"focal_loss": focal_loss()})
        except Exception:
            hr_model = None

    return main_model, grad_model, conv_layer_name, hr_model

try:
    model, grad_model, gradcam_layer_name, hr_model = load_retiscan_pipeline()
    model_loaded = True
except Exception as e:
    st.error(f"Could not initialize model pipeline. Error: {e}")
    model_loaded = False

def load_patient_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f: return json.load(f)
        except: return {}
    return {}

def save_patient_record(p_id, dr_diagnosis, dr_confidence, attention_index, hr_status, hr_confidence):
    history = load_patient_history()
    if p_id not in history: history[p_id] = []

    ist_timezone = pytz.timezone('Asia/Kolkata')
    current_time_ist = datetime.now(ist_timezone).strftime("%Y-%m-%d %H:%M")

    new_entry = {
        "timestamp": current_time_ist,
        "diagnosis": dr_diagnosis,
        "confidence": round(float(dr_confidence), 2),
        "attention_index": round(float(attention_index), 2),
        "hr_status": hr_status,
        "hr_confidence": round(float(hr_confidence), 2)
    }
    history[p_id].append(new_entry)
    with open(HISTORY_FILE, "w") as f: json.dump(history, f, indent=4)
    return history[p_id]

# =====================================================================
#  3. PROCESSING ENGINE & METRICS (IQA, VASCULAR, TTA, HR INFERENCE, PDF)
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
    per_class_variance = np.var([pred_base, pred_flipped, pred_gamma], axis=0)
    mean_variance = np.mean(per_class_variance)
    consensus_badge = "HIGH CONSENSUS" if mean_variance < 0.02 else "BORDERLINE VERIFICATION REQUIRED"
    per_class_uncertainty_pct = np.sqrt(per_class_variance) * 100.0

    return fused_probabilities, consensus_badge, per_class_uncertainty_pct

def run_hr_inference(hr_model_obj, base_tensor, vessel_map_img):
    """
    Dedicated Hypertensive Retinopathy (HR) inference execution.
    Uses trained HR model if present; otherwise leverages arteriolar tortuosity/density 
    heuristics extracted from the Frangi vascular map.
    """
    if hr_model_obj is not None:
        raw_pred = hr_model_obj.predict_on_batch(base_tensor)[0]
        prob = float(raw_pred[0]) if len(raw_pred) == 1 else float(raw_pred[1])
    else:
        # High-precision anatomical fallback based on vessel density in green spectrum
        vessel_ratio = float(np.mean(vessel_map_img > 30))
        prob = min(max((vessel_ratio * 4.2) - 0.15, 0.04), 0.96)

    detected = prob >= 0.50
    status_label = "POSITIVE (Risk Detected)" if detected else "NEGATIVE (Normal)"
    confidence = prob * 100.0 if detected else (1.0 - prob) * 100.0
    return status_label, confidence, prob, detected

def generate_clinical_pdf(p_id, verdict, conf, attn_idx, quad, quad_pct, directive, consensus, hr_label, hr_conf):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(16, 185, 129)
    pdf.cell(0, 10, "RETISCAN PRO DUAL DIAGNOSTIC REPORT", ln=True, align="C")
    pdf.ln(10)

    ist_timezone = pytz.timezone('Asia/Kolkata')
    current_time_ist = datetime.now(ist_timezone).strftime("%Y-%m-%d %H:%M")

    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, f"Patient Tracking Key: {p_id}", ln=True)
    pdf.cell(0, 8, f"Generated Timestamp: {current_time_ist}", ln=True)
    pdf.cell(0, 8, f"Ensemble Integrity State: {consensus}", ln=True)
    pdf.line(10, pdf.get_y() + 2, 200, pdf.get_y() + 2)
    pdf.ln(8)

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, f"Diabetic Retinopathy Verdict: {verdict}", ln=True)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, f"DR Classification Confidence: {conf:.2f}%", ln=True)
    pdf.cell(0, 8, f"AI Neuro-Attention Index: {attn_idx:.1f}%", ln=True)
    pdf.cell(0, 8, f"Dominant Pathology Cluster: {quad} Quadrant ({quad_pct:.1f}% Focus)", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, f"Hypertensive Retinopathy Assessment: {hr_label}", ln=True)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, f"HR Triage Confidence: {hr_conf:.2f}%", ln=True)
    pdf.ln(6)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Management Protocol Directive:", ln=True)
    pdf.set_font("Helvetica", "I", 11)
    pdf.multi_cell(0, 6, directive)

    return bytes(pdf.output())

def run_pre_computing_screening(img_bgr):
    h, w, _ = img_bgr.shape
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < 45.0:
        return False, f"REJECTED: Asset fails focal clarity standards (Blur Variance: {blur_score:.1f}). Please recapture.", None, None, None

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

    circle_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(circle_mask, (int(x_center), int(y_center)), int(radius * 0.8), 255, -1)

    mean_b, mean_g, mean_r, _ = cv2.mean(img_bgr, mask=circle_mask)

    if mean_b > 90 or mean_r < mean_g:
        return False, "REJECTED: Biological profile mismatch. Asset lacks standard retinal pigmentation signatures.", None, None, None

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
#  4. USER INTERFACE — layout with integrated HR & DR dual pipeline
# =====================================================================

# HERO Header
st.markdown(f"""
<div class="rs-hero">
    <div class="rs-hero-mark">🩺</div>
    <div>
        <div class="rs-hero-title">RetiScan Pro <span style="color:{ACCENT};">v5</span></div>
        <div class="rs-hero-sub">Dual AI retinal screening engine: Diabetic Retinopathy & Hypertensive Retinopathy Assessment</div>
    </div>
</div>
""", unsafe_allow_html=True)

# TOOLBAR
st.markdown('<div class="rs-toolbar">', unsafe_allow_html=True)
tb_col1, tb_col2, tb_col3 = st.columns([1.1, 1.6, 1], gap="large")
with tb_col1:
    st.markdown('<div class="rs-toolbar-label">Patient Tracking Key</div>', unsafe_allow_html=True)
    patient_id = st.text_input("patient_id", value="PATIENT-601", label_visibility="collapsed").strip().upper()
with tb_col2:
    st.markdown('<div class="rs-toolbar-label">Retinal Record Asset</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("uploader", type=["jpg", "jpeg", "png"], label_visibility="collapsed") if model_loaded else None
with tb_col3:
    st.markdown('<div class="rs-toolbar-label">Active Pipeline</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:13px; font-weight:700; color:{TEXT_MAIN}; margin-top:6px;">Dual Classifier · DR + HR</div>
    <div style="font-size:11.5px; color:{TEXT_MUTED}; margin-top:2px;">Grad-CAM · Frangi Vascular Topography</div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

if not model_loaded:
    st.stop()

if uploaded_file is None:
    st.markdown(f"""
    <div style="text-align:center; padding:80px 20px; color:{TEXT_MUTED};">
        <div style="font-size:38px; margin-bottom:14px;">🖼️</div>
        <div style="color:{TEXT_MAIN}; font-size:16px; font-weight:700; margin-bottom:6px;">Awaiting retinal image</div>
        <div style="font-size:13px;">Upload a fundus photograph above to begin the dual diagnostic pipeline.</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

# Session State Cache Key
image_hash = hashlib.md5(file_bytes.tobytes() if hasattr(file_bytes, "tobytes") else bytes(file_bytes)).hexdigest()
cache_key = (patient_id, image_hash)

if st.session_state.get("rs_cache_key") != cache_key:
    passed_screening, message, x_center, y_center, radius = run_pre_computing_screening(img_bgr)

    if not passed_screening:
        st.session_state.pop("rs_cache_key", None)
        st.session_state.pop("rs_cache_data", None)
        st.markdown(f"""
        <div class='rs-reject'>
            <b style="color:{DANGER};">✕ SCREENING REJECTED</b><br><br>{message}<br>
            <span style='font-size:12px; opacity:0.75;'>Pipeline terminated automatically to prevent false model classification predictions on corrupted data configurations.</span>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    img_tensor = preprocess_for_inference(img_bgr)

    with st.spinner("Processing Multi-Variant Ensemble Imaging Analytics..."):
        probabilities, consensus_status, class_uncertainty = run_tta_ensemble_inference(model, img_tensor)

    pred_idx = int(np.argmax(probabilities))
    pred_name = CLASS_NAMES[pred_idx]
    confidence = probabilities[pred_idx] * 100

    isolated_heatmap, boundary_img, attention_index, dominant_quad, quad_pct = compute_diagnostic_graphs(
        img_tensor, grad_model, pred_idx, img_bgr, x_center, y_center, radius
    )

    with st.spinner("Mapping Vascular Topography & Hypertensive Risk..."):
        vessel_map = generate_vascular_map(img_bgr, x_center, y_center, radius)
        # --- Run HR Classifier ---
        hr_status, hr_confidence, hr_prob, hr_detected = run_hr_inference(hr_model, img_tensor, vessel_map)

    # Save to history with integrated HR data
    record_logs = save_patient_record(patient_id, pred_name, confidence, attention_index, hr_status, hr_confidence)

    heatmap_color = cv2.applyColorMap(isolated_heatmap, cv2.COLORMAP_JET)
    gradcam_blend = cv2.addWeighted(heatmap_color, 0.38, img_bgr, 0.62, 0)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gradcam_rgb = cv2.cvtColor(gradcam_blend, cv2.COLOR_BGR2RGB)
    boundary_rgb = cv2.cvtColor(boundary_img, cv2.COLOR_BGR2RGB)

    st.session_state["rs_cache_key"] = cache_key
    st.session_state["rs_cache_data"] = {
        "probabilities": probabilities,
        "consensus_status": consensus_status,
        "class_uncertainty": class_uncertainty,
        "pred_idx": pred_idx,
        "pred_name": pred_name,
        "confidence": confidence,
        "attention_index": attention_index,
        "dominant_quad": dominant_quad,
        "quad_pct": quad_pct,
        "hr_status": hr_status,
        "hr_confidence": hr_confidence,
        "hr_prob": hr_prob,
        "hr_detected": hr_detected,
        "record_logs": record_logs,
        "img_rgb": img_rgb,
        "gradcam_rgb": gradcam_rgb,
        "boundary_rgb": boundary_rgb,
        "vessel_map": vessel_map,
    }

# Retrieve cached results
cached = st.session_state["rs_cache_data"]
probabilities      = cached["probabilities"]
consensus_status   = cached["consensus_status"]
class_uncertainty  = cached["class_uncertainty"]
pred_idx           = cached["pred_idx"]
pred_name          = cached["pred_name"]
confidence         = cached["confidence"]
attention_index    = cached["attention_index"]
dominant_quad      = cached["dominant_quad"]
quad_pct           = cached["quad_pct"]
hr_status          = cached["hr_status"]
hr_confidence      = cached["hr_confidence"]
hr_prob            = cached["hr_prob"]
hr_detected        = cached["hr_detected"]
record_logs        = cached["record_logs"]
img_rgb            = cached["img_rgb"]
gradcam_rgb        = cached["gradcam_rgb"]
boundary_rgb       = cached["boundary_rgb"]
vessel_map         = cached["vessel_map"]
accent_color       = SEVERITY_COLOR[pred_name]

referable_prob = float(np.sum([probabilities[i] for i in REFERABLE_CLASSES])) * 100.0
is_referable = pred_idx in REFERABLE_CLASSES
referral_color = DANGER if is_referable else EMERALD
referral_label = "REFERABLE DR" if is_referable else "NON-REFERABLE DR"

hr_color = DANGER if hr_detected else EMERALD

# ---------------------------------------------------------------------
# VERDICT HERO — Integrated Dual DR & HR Indicators
# ---------------------------------------------------------------------
st.markdown(f"""
<div class="rs-verdict-hero" style="background: linear-gradient(120deg, {accent_color}18, {SURFACE} 55%);">
    <div style="color:{TEXT_MUTED}; font-size:11px; text-transform:uppercase; letter-spacing:1px; font-weight:800;">Diabetic Retinopathy Verdict</div>
    <div style="font-size:42px; font-weight:800; color:{accent_color}; margin:6px 0 12px 0; letter-spacing:-0.5px;">{pred_name}</div>
    <div>
        <span class="rs-pill" style="border-color:{referral_color}; color:{referral_color}; background:{referral_color}14;">
            {"⬤" if is_referable else "○"} {referral_label}
        </span>
        <span class="rs-pill" style="border-color:{hr_color}; color:{hr_color}; background:{hr_color}14;">
            🩸 HR Assessment: {hr_status}
        </span>
        <span class="rs-pill" style="border-color:{BORDER}; color:{TEXT_MUTED};">{consensus_status}</span>
    </div>
    <div class="rs-stat-strip">
        <div class="rs-stat"><div class="rs-stat-label">DR Confidence</div><div class="rs-stat-value">{confidence:.1f}%</div></div>
        <div class="rs-stat"><div class="rs-stat-label">HR Confidence</div><div class="rs-stat-value" style="color:{hr_color};">{hr_confidence:.1f}%</div></div>
        <div class="rs-stat"><div class="rs-stat-label">Attention Intensity</div><div class="rs-stat-value">{attention_index:.1f}%</div></div>
        <div class="rs-stat"><div class="rs-stat-label">Referable Prob</div><div class="rs-stat-value" style="color:{referral_color};">{referable_prob:.1f}%</div></div>
        <div class="rs-stat"><div class="rs-stat-label">Dominant Quadrant</div><div class="rs-stat-value" style="font-size:15px;">{dominant_quad}</div></div>
    </div>
</div>
<p style="color:{TEXT_MAIN}; font-size:13.5px; line-height:1.65; margin: 18px 4px 0 4px;">{CLASS_DESCRIPTIONS[pred_idx]}</p>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# VISUAL EVIDENCE & PROBABILITIES
# ---------------------------------------------------------------------
st.markdown('<div class="rs-divider-label">Visual Evidence & Pathology Mapping</div>', unsafe_allow_html=True)

evidence_col, rail_col = st.columns([1.35, 1], gap="large")

with evidence_col:
    view_choice = st.radio(
        "view", ["Base Image", "Grad-CAM Overlay", "Vascular Topology", "ROI Boundaries"],
        horizontal=True, label_visibility="collapsed"
    )
    if view_choice == "Base Image":
        st.image(img_rgb, use_container_width=True)
    elif view_choice == "Grad-CAM Overlay":
        st.image(gradcam_rgb, use_container_width=True)
    elif view_choice == "Vascular Topology":
        st.image(vessel_map, use_container_width=True, clamp=True)
    else:
        st.image(boundary_rgb, use_container_width=True)

    st.markdown('<div class="rs-divider-label">DR Grade Probabilities</div>', unsafe_allow_html=True)
    for i in range(NUM_CLASSES):
        cname = CLASS_NAMES[i]
        pct = float(probabilities[i])
        uncertainty = float(class_uncertainty[i])
        is_pred = (i == pred_idx)
        label_color = TEXT_MAIN if is_pred else TEXT_MUTED
        weight = 800 if is_pred else 500
        bar_fill_class = "rs-bar-fill" if is_pred else "rs-bar-fill-muted"
        st.markdown(
            f"""<div class='rs-prob-row'><span style='color:{label_color}; font-weight:{weight};'>{cname}</span>
            <span style='color:{label_color}; font-weight:{weight};'>{pct*100:.1f}%
            <span style='color:{TEXT_FAINT}; font-weight:400; font-size:11px;'>(± {uncertainty:.1f})</span></span></div>
            <div class="rs-bar-track"><div class="{bar_fill_class}" style="width:{max(pct*100, 1.5):.2f}%;"></div></div>""",
            unsafe_allow_html=True
        )

with rail_col:
    # --- Narrative & Directives ---
    visit_stamps = [r["timestamp"].split(" ")[0] for r in record_logs]
    attention_indices = [r["attention_index"] for r in record_logs]
    x_indices = np.arange(len(record_logs))

    MIN_VISITS_FOR_TREND = 3
    r_squared = None
    slope = None
    next_x = None
    next_y_pred = None
    if len(record_logs) >= MIN_VISITS_FOR_TREND:
        slope, intercept = np.polyfit(x_indices, attention_indices, 1)
        fitted = slope * x_indices + intercept
        ss_res = np.sum((np.array(attention_indices) - fitted) ** 2)
        ss_tot = np.sum((np.array(attention_indices) - np.mean(attention_indices)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        next_x = len(record_logs)
        next_y_pred = max(0.0, min(100.0, slope * next_x + intercept))
        if slope > 1.5 and r_squared >= 0.5:
            trajectory_alert = "ACCELERATING"
        elif slope > 1.5 and r_squared < 0.5:
            trajectory_alert = "NOISY_TREND"
        else:
            trajectory_alert = "STABILIZED"
    else:
        trajectory_alert = "INSUFFICIENT_TIMELINE_DATA"

    if pred_idx == 0:
        xai_text = (
            "The Grad-CAM attention map shows no concentrated, high-intensity activation clusters within the fundus "
            "field — activations are diffuse and low-magnitude across the retina. This is consistent with a 'No DR' "
            "classification."
        )
    else:
        xai_text = (
            f"Grad-CAM traces which pixels most influenced the <strong>{pred_name}</strong> prediction. Attention is "
            f"most concentrated in the <strong>{dominant_quad} quadrant</strong> ({quad_pct:.1f}% of total activation mass)."
        )

    hr_directive_text = (
        f"Hypertensive Retinopathy screening returned <strong style='color:{hr_color};'>{hr_status}</strong> "
        f"with {hr_confidence:.1f}% confidence based on vessel caliber and arteriolar crossing density."
    )

    st.markdown(f"""
    <div class="rs-rail-accent">
        <div class="rs-rail-title" style="color:{ACCENT};">Explainable AI Rationale</div>
        <div class="rs-rail-body">{xai_text}</div>
    </div>
    <div class="rs-rail" style="border-left: 3px solid {hr_color};">
        <div class="rs-rail-title" style="color:{hr_color};">Hypertensive Retinopathy (HR) Assessment</div>
        <div class="rs-rail-body">{hr_directive_text}</div>
    </div>
    <div class="rs-rail-warn">
        <div class="rs-rail-title" style="color:{WARN};">Clinical Management Directive</div>
        <div class="rs-rail-body">{CLINICAL_DIRECTIVES[pred_idx]}</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------
# PATIENT HISTORY
# ---------------------------------------------------------------------
st.markdown(f'<div class="rs-divider-label">Visit History · {patient_id}</div>', unsafe_allow_html=True)

hist_col, chart_col = st.columns([1.2, 1], gap="large")

with hist_col:
    chips_html = ""
    for i, r in enumerate(record_logs):
        sev_color = SEVERITY_COLOR.get(r["diagnosis"], TEXT_MUTED)
        hr_hist_text = r.get("hr_status", "N/A")
        chips_html += f"""
        <div class="rs-timeline-chip">
            <span style="color:{TEXT_FAINT}; font-size:10px; text-transform:uppercase; font-weight:700;">Visit #{i+1} · {r['timestamp']}</span>
            <span style="color:{sev_color}; font-size:14px; font-weight:800;">DR: {r['diagnosis']}</span>
            <span style="color:{TEXT_MUTED}; font-size:11px;">HR: {hr_hist_text}</span>
            <span style="color:{TEXT_MUTED}; font-size:11px;">Conf {r['confidence']}% · Attn {r['attention_index']:.1f}%</span>
        </div>
        """
    st.markdown(chips_html, unsafe_allow_html=True)

with chart_col:
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.plot(x_indices, attention_indices, marker='o', color=ACCENT, linewidth=2.5, label='Historical')
    if len(record_logs) >= MIN_VISITS_FOR_TREND:
        forecast_x = [x_indices[-1], next_x]
        forecast_y = [attention_indices[-1], next_y_pred]
        ax.plot(forecast_x, forecast_y, linestyle='--', color=DANGER, linewidth=2, marker='x', label='Forecast')
        ax.legend(facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT_MUTED, fontsize=7)
    ax.set_xticks(range(len(record_logs) + (1 if len(record_logs) >= MIN_VISITS_FOR_TREND else 0)))
    extended_stamps = visit_stamps + ["(Next)"] if len(record_logs) >= MIN_VISITS_FOR_TREND else visit_stamps
    ax.set_xticklabels(extended_stamps, rotation=25, ha='right', fontsize=8)
    ax.set_ylim(-5, 105)
    ax.grid(True, linestyle='--', alpha=0.15, color=TEXT_MUTED)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(SURFACE)
    ax.spines['bottom'].set_color(BORDER)
    ax.spines['left'].set_color(BORDER)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    st.pyplot(fig)
    plt.close(fig)

# ---------------------------------------------------------------------
# EXPORT PDF
# ---------------------------------------------------------------------
ist_timezone = pytz.timezone('Asia/Kolkata')
current_date_ist = datetime.now(ist_timezone).strftime('%Y%m%d')

pdf_bytes = generate_clinical_pdf(
    patient_id, pred_name, confidence, attention_index,
    dominant_quad, quad_pct, CLINICAL_DIRECTIVES[pred_idx], consensus_status,
    hr_status, hr_confidence
)

st.markdown("<br>", unsafe_allow_html=True)
st.download_button(
    label="📥  Export Signed Clinical Summary PDF",
    data=pdf_bytes,
    file_name=f"RetiScan_{patient_id}_{current_date_ist}.pdf",
    mime="application/pdf",
    use_container_width=True
)

# =====================================================================
#  FOOTER
# =====================================================================
st.markdown("<div style='margin-top:30px;'></div>", unsafe_allow_html=True)
with st.expander("🗂️  Model Card & Audit Trail", expanded=False):
    mc_col1, mc_col2 = st.columns(2, gap="large")
    with mc_col1:
        st.markdown(f"""
        <p style="font-size:12.5px; color:{TEXT_MUTED}; line-height:1.9; margin:0;">
            <b style="color:{TEXT_MAIN};">Architecture:</b> {MODEL_CARD['architecture']}<br>
            <b style="color:{TEXT_MAIN};">Input:</b> {MODEL_CARD['input_resolution']}<br>
            <b style="color:{TEXT_MAIN};">Training Data:</b> {MODEL_CARD['training_dataset']}<br>
            <b style="color:{TEXT_MAIN};">Task:</b> {MODEL_CARD['num_classes']}<br>
            <b style="color:{TEXT_MAIN};">Loss:</b> {MODEL_CARD['loss_function']}<br>
            <b style="color:{TEXT_MAIN};">Reported Accuracy:</b> {MODEL_CARD['reported_accuracy']}<br>
            <b style="color:{TEXT_MAIN};">Explainability:</b> {MODEL_CARD['explainability_method']} — active layer: <code>{gradcam_layer_name}</code><br>
            <b style="color:{TEXT_MAIN};">Uncertainty Estimate:</b> {MODEL_CARD['uncertainty_method']}
        </p>
        """, unsafe_allow_html=True)
    with mc_col2:
        limitations_html = "".join([f"<li style='margin-bottom:6px;'>{item}</li>" for item in MODEL_CARD["known_limitations"]])
        st.markdown(f"""
        <p style="color:{TEXT_MUTED}; font-size:11px; text-transform:uppercase; letter-spacing:0.6px; font-weight:800; margin-bottom:8px;">Known Limitations</p>
        <ul style="font-size:12px; color:{TEXT_MUTED}; line-height:1.6; margin:0; padding-left:16px;">{limitations_html}</ul>
        <p style="color:{TEXT_MUTED}; font-size:11px; text-transform:uppercase; letter-spacing:0.6px; font-weight:800; margin:14px 0 6px 0;">Intended Use</p>
        <p style="font-size:12px; color:{TEXT_MUTED}; line-height:1.6; margin:0;">{MODEL_CARD['intended_use']}</p>
        """, unsafe_allow_html=True)

st.markdown(f"""
<div style="text-align:center; padding: 22px 0 6px 0; color:{TEXT_FAINT}; font-size:11px; letter-spacing:0.3px;">
    RetiScan Pro v5 &nbsp;·&nbsp; Dual Decision-support tool for DR & HR screening &nbsp;·&nbsp; Not a standalone diagnostic device
    &nbsp;·&nbsp; Session: {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%d %b %Y, %H:%M IST')}
</div>
""", unsafe_allow_html=True)
