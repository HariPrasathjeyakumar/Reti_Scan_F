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
from skimage.filters import frangi  # Mathematical vascular extraction

# Set page configurations to clean wide layout
st.set_page_config(
    page_title="RetiScan Pro v5 Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
#  0. DESIGN TOKENS — single source of truth for the clinical theme
# =====================================================================
# Deep clinical-tech theme: navy/slate surfaces, high-contrast text,
# a single confident accent color, and semantic status colors.
BG          = "#0c1220"
SURFACE     = "#141b2d"
SURFACE_ALT = "#1b2438"
SURFACE_RAISED = "#202b44"
BORDER      = "#2b3652"
TEXT_MAIN   = "#eef1f8"
TEXT_MUTED  = "#8d96af"
ACCENT      = "#22d3c8"     # teal-cyan — primary clinical accent
ACCENT_DARK = "#14a99f"
ACCENT_SOFT = "#173634"
INFO        = "#4f9bff"
WARN        = "#f0b23c"
WARN_BG     = "#332612"
WARN_BORDER = "#584419"
DANGER      = "#ff5c5c"
DANGER_BG   = "#301416"
DANGER_BORDER = "#5c2226"
SUCCESS     = "#33d183"
SUCCESS_BG  = "#122a20"
SUCCESS_BORDER = "#1e4d38"
XAI_BG      = "#141d3a"
XAI_BORDER  = "#2c3f6e"
XAI_TEXT    = "#8fb4ff"

st.markdown(f"""
    <style>
    /* ---------- Global canvas ---------- */
    html, body, .main, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
        background-color: {BG} !important;
        color: {TEXT_MAIN};
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    [data-testid="stHeader"] {{
        background-color: {BG} !important;
    }}
    .block-container {{
        padding-top: 1.6rem;
        padding-bottom: 3rem;
        max-width: 1300px;
    }}

    /* ---------- Sidebar ---------- */
    div[data-testid="stSidebar"] {{
        background-color: {SURFACE};
        border-right: 1px solid {BORDER};
    }}
    div[data-testid="stSidebar"] * {{ color: {TEXT_MAIN}; }}
    div[data-testid="stSidebar"] label {{
        font-weight: 700;
        font-size: 11.5px;
        color: {TEXT_MUTED} !important;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }}

    /* ---------- Inputs ---------- */
    .stTextInput input {{
        background-color: {SURFACE_ALT} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 8px !important;
        color: {TEXT_MAIN} !important;
    }}
    .stTextInput input:focus {{
        border-color: {ACCENT} !important;
        box-shadow: 0 0 0 1px {ACCENT} !important;
    }}

    /* ---------- File uploader (fixes "fully white" button issue) ---------- */
    [data-testid="stFileUploaderDropzone"] {{
        background-color: {SURFACE_ALT} !important;
        border: 1.5px dashed {BORDER} !important;
        border-radius: 10px !important;
    }}
    [data-testid="stFileUploaderDropzone"] * {{
        color: {TEXT_MUTED} !important;
    }}
    [data-testid="stFileUploaderDropzone"] button,
    [data-testid="baseButton-secondary"],
    section[data-testid="stFileUploader"] button {{
        background-color: {SURFACE_RAISED} !important;
        color: {TEXT_MAIN} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }}
    [data-testid="stFileUploaderDropzone"] button:hover,
    section[data-testid="stFileUploader"] button:hover {{
        border-color: {ACCENT} !important;
        color: {ACCENT} !important;
    }}
    [data-testid="stFileUploaderFile"] {{
        background-color: {SURFACE_RAISED} !important;
        border-radius: 8px !important;
        border: 1px solid {BORDER} !important;
    }}

    /* ---------- All generic buttons (kind=primary/secondary) ---------- */
    button[kind="primary"], button[kind="secondary"],
    .stButton button, .stDownloadButton button {{
        background-color: {ACCENT} !important;
        color: #06201d !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        padding: 10px 16px !important;
        box-shadow: 0 2px 10px rgba(34, 211, 200, 0.25);
        transition: all 0.15s ease-in-out;
    }}
    .stButton button:hover, .stDownloadButton button:hover {{
        background-color: {ACCENT_DARK} !important;
        box-shadow: 0 2px 14px rgba(34, 211, 200, 0.4);
    }}
    .stDownloadButton button p {{ color: #06201d !important; font-weight: 700 !important; }}

    /* ---------- Expander ---------- */
    .stExpander {{
        background-color: {SURFACE} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 12px;
        margin-top: 15px;
    }}
    .stExpander summary {{
        color: {TEXT_MAIN} !important;
        font-weight: 600 !important;
    }}

    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        border-bottom: 1px solid {BORDER};
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: transparent;
        color: {TEXT_MUTED};
        font-weight: 600;
        font-size: 13.5px;
        border-radius: 8px 8px 0 0;
        padding: 10px 18px;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {SURFACE} !important;
        color: {ACCENT} !important;
        border-bottom: 2px solid {ACCENT} !important;
    }}

    /* ---------- Misc Streamlit widgets ---------- */
    div[data-testid="stBlock"] {{ border-radius: 12px; }}
    [data-testid="stMetricValue"] {{ color: {TEXT_MAIN}; }}
    [data-testid="stMetricLabel"] {{ color: {TEXT_MUTED}; }}
    .stProgress > div > div {{ background-color: {ACCENT} !important; }}
    .stProgress > div {{ background-color: {SURFACE_ALT} !important; }}
    hr {{ border-color: {BORDER} !important; }}
    ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
    ::-webkit-scrollbar-thumb {{ background-color: {BORDER}; border-radius: 8px; }}

    /* ---------- Custom component classes ---------- */
    .rs-topbar {{
        display: flex; align-items: center; justify-content: space-between;
        padding-bottom: 18px; margin-bottom: 22px; border-bottom: 1px solid {BORDER};
    }}
    .rs-brand {{ display: flex; align-items: center; gap: 14px; }}
    .rs-brand-icon {{
        width: 46px; height: 46px; border-radius: 12px;
        background: linear-gradient(135deg, {ACCENT}, {ACCENT_DARK});
        display: flex; align-items: center; justify-content: center; font-size: 22px;
    }}
    .rs-kpi {{
        background-color: {SURFACE}; border: 1px solid {BORDER}; border-radius: 12px;
        padding: 16px 18px; height: 100%;
    }}
    .rs-kpi-label {{
        color: {TEXT_MUTED}; font-size: 10.5px; text-transform: uppercase;
        letter-spacing: 0.6px; font-weight: 700; margin-bottom: 6px;
    }}
    .rs-kpi-value {{ font-size: 22px; font-weight: 700; line-height: 1.2; }}
    .rs-section-label {{
        color: {TEXT_MAIN}; font-size: 13px; text-transform: uppercase;
        letter-spacing: 1px; font-weight: 700; margin-bottom: 16px;
        border-left: 4px solid {ACCENT}; padding-left: 10px;
    }}
    .rs-subtle-label {{
        color: {TEXT_MUTED}; font-size: 11px; text-transform: uppercase;
        letter-spacing: 0.6px; font-weight: 700;
    }}
    .rs-subtle-label-accent {{
        color: {ACCENT}; font-size: 11px; text-transform: uppercase;
        letter-spacing: 0.6px; font-weight: 700;
    }}
    .rs-card {{
        background-color: {SURFACE}; padding: 26px; border-radius: 14px;
        border: 1px solid {BORDER}; height: 100%;
    }}
    .rs-footer-card {{
        background-color: {SURFACE_ALT}; padding: 20px; border-radius: 12px;
        border: 1px solid {BORDER}; height: 100%;
    }}
    .rs-banner {{
        border-radius: 12px; padding: 18px 22px; display: flex; gap: 14px;
        margin-bottom: 16px; border: 1px solid {BORDER};
    }}
    .rs-banner-title {{
        font-size: 11.5px; text-transform: uppercase; letter-spacing: 1px;
        font-weight: 700; margin-bottom: 6px;
    }}
    .rs-banner-body {{ font-size: 13.5px; line-height: 1.65; }}
    .rs-reject {{
        background: {DANGER_BG}; color: #ffb4b4; padding: 18px 22px;
        border-radius: 12px; border: 1px solid {DANGER_BORDER}; margin-top: 15px;
    }}
    .rs-table {{
        width: 100%; text-align: left; border-collapse: collapse; font-size: 13.5px;
        background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px; overflow: hidden;
    }}
    .rs-table th {{
        padding: 12px 14px; color: {TEXT_MUTED}; background: {SURFACE_ALT};
        border-bottom: 1px solid {BORDER}; font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.5px;
    }}
    .rs-table td {{ padding: 12px 14px; border-bottom: 1px solid {BORDER}; color: {TEXT_MAIN}; }}
    .rs-badge {{
        display:inline-block; font-size:11.5px; font-weight:700; padding:5px 10px;
        border-radius:6px; border:1px solid;
    }}
    </style>
""", unsafe_allow_html=True)

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
    "No DR":            "#158a5c",
    "Mild NPDR":         "#b9781a",
    "Moderate NPDR":     "#1c6fd6",
    "Severe NPDR":       "#d0651f",
    "Proliferative DR":  "#c0271f",
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

# Clinical referral mapping — Moderate NPDR and above requires specialist referral
# (standard binary screening convention used alongside 5-class ICDR grading)
REFERABLE_CLASSES = {2, 3, 4}
NON_REFERABLE_CLASSES = {0, 1}

MODEL_CARD = {
    "architecture": "EfficientNetB3 (ImageNet-pretrained backbone, fine-tuned)",
    "input_resolution": f"{IMG_SIZE} x {IMG_SIZE} RGB",
    "training_dataset": "APTOS 2019 Blindness Detection dataset",
    "num_classes": "5-class ICDR severity scale (No DR → Proliferative DR)",
    "loss_function": "Categorical Crossentropy with label smoothing (focal-loss variant explored)",
    "reported_accuracy": "~89% top-1 accuracy on held-out validation split (5-class grading)",
    "explainability_method": "Grad-CAM (gradient-weighted class activation mapping), single convolutional layer",
    "uncertainty_method": "Test-Time Augmentation (TTA) ensemble variance across 3 augmented views (identity, horizontal flip, gamma shift)",
    "known_limitations": [
        "Trained and validated on a single dataset (APTOS 2019); performance on images from other cameras/populations has not been externally verified.",
        "Grad-CAM highlights regions correlated with the prediction — it does not guarantee the highlighted region is the true clinical lesion.",
        "Longitudinal trend forecasting requires at least 3 visits to be statistically meaningful; earlier visits are descriptive only.",
        "The model has not been evaluated for calibration (i.e. whether a 90% confidence score genuinely corresponds to 90% real-world accuracy).",
        "Not a diagnostic device. Intended as a decision-support and triage aid alongside qualified clinical judgement.",
    ],
    "intended_use": "Screening triage support for diabetic retinopathy grading, to prioritize specialist review — not a replacement for ophthalmologist diagnosis.",
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
    return main_model, grad_model, conv_layer_name

try:
    model, grad_model, gradcam_layer_name = load_retiscan_pipeline()
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

    # Timezone alignment: Force conversion to Indian Standard Time (IST)
    ist_timezone = pytz.timezone('Asia/Kolkata')
    current_time_ist = datetime.now(ist_timezone).strftime("%Y-%m-%d %H:%M")

    new_entry = {
        "timestamp": current_time_ist,
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

    per_class_variance = np.var([pred_base, pred_flipped, pred_gamma], axis=0)
    mean_variance = np.mean(per_class_variance)
    consensus_badge = "HIGH CONSENSUS" if mean_variance < 0.02 else "BORDERLINE VERIFICATION REQUIRED"

    # Per-class uncertainty band expressed as +/- percentage points (std dev across TTA views)
    per_class_uncertainty_pct = np.sqrt(per_class_variance) * 100.0

    return fused_probabilities, consensus_badge, per_class_uncertainty_pct

def generate_clinical_pdf(p_id, verdict, conf, attn_idx, quad, quad_pct, directive, consensus):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(16, 185, 129)
    pdf.cell(0, 10, "RETISCAN PRO DIAGNOSTIC SUMMARY REPORT", ln=True, align="C")
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
st.markdown(f"""
<div class="rs-topbar">
    <div class="rs-brand">
        <div class="rs-brand-icon">🩺</div>
        <div>
            <div style="color:{TEXT_MAIN}; font-size:24px; font-weight:700; line-height:1.2;">RetiScan Pro <span style="color:{ACCENT};">v5</span></div>
            <div style="color:{TEXT_MUTED}; font-size:12.5px; letter-spacing:0.3px;">Diabetic Retinopathy Grading &amp; Explainable AI Dashboard</div>
        </div>
    </div>
    <div style="text-align:right;">
        <div style="color:{TEXT_MUTED}; font-size:10.5px; text-transform:uppercase; letter-spacing:0.6px; font-weight:700;">Screening Session</div>
        <div style="color:{TEXT_MAIN}; font-size:13px; font-weight:600;">{datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%d %b %Y, %H:%M IST')}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Sidebar — Clinical Control Hub
# ---------------------------------------------------------------------
st.sidebar.markdown(f"""
<div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
    <span style="font-size:20px;">🧭</span>
    <span style="color:{TEXT_MAIN}; font-weight:700; font-size:16px;">Clinical Control Hub</span>
</div>
<div style="color:{TEXT_MUTED}; font-size:11.5px; margin-bottom:20px;">Patient intake &amp; imaging upload</div>
""", unsafe_allow_html=True)
patient_id = st.sidebar.text_input("Patient ID Key", value="PATIENT-601").strip().upper()

if model_loaded:
    uploaded_file = st.sidebar.file_uploader("Upload Retinal Record Asset", type=["jpg", "jpeg", "png"])
    st.sidebar.markdown(f"""
    <div style="margin-top:24px; padding:14px; background:{SURFACE_ALT}; border:1px solid {BORDER}; border-radius:10px;">
        <div style="color:{TEXT_MUTED}; font-size:10.5px; text-transform:uppercase; letter-spacing:0.5px; font-weight:700; margin-bottom:6px;">Active Model</div>
        <div style="color:{TEXT_MAIN}; font-size:12.5px; font-weight:600;">EfficientNetB3 · 5-class ICDR</div>
        <div style="color:{TEXT_MUTED}; font-size:11px; margin-top:4px;">Grad-CAM · TTA Ensemble Uncertainty</div>
    </div>
    """, unsafe_allow_html=True)

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        passed_screening, message, x_center, y_center, radius = run_pre_computing_screening(img_bgr)

        if not passed_screening:
            st.markdown(f"""
            <div class='rs-reject'>
                <b>❌ SYSTEM SCREENING REJECTION</b><br>{message}<br>
                <span style='font-size:12px; font-weight:normal; opacity:0.8;'>Pipeline terminated automatically to prevent false model classification predictions on corrupted data configurations.</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            img_tensor = preprocess_for_inference(img_bgr)

            with st.spinner("Processing Multi-Variant Ensemble Imaging Analytics..."):
                probabilities, consensus_status, class_uncertainty = run_tta_ensemble_inference(model, img_tensor)

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

            # Secondary clinical output: referable vs non-referable DR (standard screening
            # convention — Moderate NPDR and above requires specialist referral). Derived
            # directly from the same fused probabilities, no separate model needed.
            referable_prob = float(np.sum([probabilities[i] for i in REFERABLE_CLASSES])) * 100.0
            is_referable = pred_idx in REFERABLE_CLASSES
            referral_color = DANGER if is_referable else SUCCESS
            referral_label = "REFERABLE" if is_referable else "NON-REFERABLE"

            # ---------------------------------------------------------
            # TOP KPI STRIP — quick-glance summary before any detail
            # ---------------------------------------------------------
            kpi1, kpi2, kpi3, kpi4 = st.columns(4, gap="medium")
            with kpi1:
                st.markdown(f"""
                <div class="rs-kpi">
                    <div class="rs-kpi-label">Diagnostic Verdict</div>
                    <div class="rs-kpi-value" style="color:{accent_color};">{pred_name}</div>
                </div>""", unsafe_allow_html=True)
            with kpi2:
                st.markdown(f"""
                <div class="rs-kpi">
                    <div class="rs-kpi-label">Model Confidence</div>
                    <div class="rs-kpi-value" style="color:{TEXT_MAIN};">{confidence:.1f}%</div>
                </div>""", unsafe_allow_html=True)
            with kpi3:
                st.markdown(f"""
                <div class="rs-kpi">
                    <div class="rs-kpi-label">Referral Status</div>
                    <div class="rs-kpi-value" style="color:{referral_color}; font-size:17px;">{referral_label}</div>
                </div>""", unsafe_allow_html=True)
            with kpi4:
                st.markdown(f"""
                <div class="rs-kpi">
                    <div class="rs-kpi-label">Ensemble Consensus</div>
                    <div class="rs-kpi-value" style="color:{INFO}; font-size:15px;">{consensus_status}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ---------------------------------------------------------
            # TABBED WORKSPACE — restructures the previous single long
            # scroll into focused views without touching any computation
            # ---------------------------------------------------------

            # ---------------------------------------------------------
            # Longitudinal trend pre-computation — done once, before tabs,
            # because both the Overview tab (XAI narrative) and the History
            # tab (forecast chart) reference these values.
            # ---------------------------------------------------------
            visit_stamps = [r["timestamp"].split(" ")[0] for r in record_logs]
            attention_indices = [r["attention_index"] for r in record_logs]
            x_indices = np.arange(len(record_logs))

            # Trend forecasting requires >=3 visits to be statistically defensible.
            # A 2-point "trend" is just a line between two dots, not evidence of velocity.
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

                # Require both a meaningful slope AND a reasonable fit before flagging acceleration
                if slope > 1.5 and r_squared >= 0.5:
                    trajectory_alert = "ACCELERATING"
                elif slope > 1.5 and r_squared < 0.5:
                    trajectory_alert = "NOISY_TREND"
                else:
                    trajectory_alert = "STABILIZED"
            else:
                trajectory_alert = "INSUFFICIENT_TIMELINE_DATA"

            tab_overview, tab_probability, tab_advanced, tab_history = st.tabs(
                ["🔬  Diagnosis Overview", "📊  Probability & Uncertainty", "🧬  Advanced Imaging", "📈  Patient History"]
            )

            # ===================== TAB 1: DIAGNOSIS OVERVIEW =====================
            with tab_overview:
                st.markdown("<div class='rs-section-label'>Primary Diagnostic Trackers</div>", unsafe_allow_html=True)
                img_col1, img_col2 = st.columns(2, gap="large")
                with img_col1:
                    st.markdown("<span class='rs-subtle-label'>I. Base Input</span>", unsafe_allow_html=True)
                    st.image(img_rgb, use_container_width=True)
                with img_col2:
                    st.markdown("<span class='rs-subtle-label-accent'>II. Grad-CAM Path Map</span>", unsafe_allow_html=True)
                    st.image(gradcam_rgb, use_container_width=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"""
                    <div class="rs-card">
                        <p style="color: {TEXT_MUTED}; font-size: 11px; text-transform: uppercase; margin: 0; letter-spacing: 0.5px;">Diagnostic Verdict</p>
                        <h1 style="color: {accent_color}; margin: 10px 0 6px 0; font-size: 34px; font-weight: 700;">{pred_name}</h1>
                        <div class="rs-badge" style="background:{referral_color}22; border-color:{referral_color}; color:{referral_color}; margin-bottom:14px;">
                            {referral_label} &nbsp;·&nbsp; {"Specialist Review Advised" if is_referable else "Routine Monitoring"} &nbsp;·&nbsp; {referable_prob:.1f}% referable-class probability
                        </div>
                        <p style="color: {TEXT_MUTED}; font-size: 12.5px; margin: 0;">Confidence: <b style="color:{TEXT_MAIN};">{confidence:.2f}%</b> &nbsp;|&nbsp; Attention Intensity Score: <b style="color:{TEXT_MAIN};">{attention_index:.1f}%</b></p>
                        <p style="color: {TEXT_MAIN}; font-size: 13.5px; margin-top: 14px; line-height: 1.6;">{CLASS_DESCRIPTIONS[pred_idx]}</p>
                    </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # --- Explainable AI narrative (unchanged logic, only container/styling changed) ---
                if pred_idx == 0:
                    xai_text = (
                        "The Grad-CAM attention map shows no concentrated, high-intensity activation clusters within the "
                        "fundus field — activations are diffuse and low-magnitude across the retina. In practice, this means "
                        "the model did not lock onto any localized region resembling a lesion, which is consistent with a "
                        "'No DR' classification rather than a positive finding the model is choosing to ignore."
                    )
                else:
                    xai_text = (
                        f"Grad-CAM traces which pixels most influenced the model's <strong>{pred_name}</strong> prediction by "
                        f"back-propagating the class score to the final convolutional layer. The resulting attention is most "
                        f"concentrated in the <strong>{dominant_quad} quadrant</strong>, accounting for {quad_pct:.1f}% of the "
                        f"total activation mass inside the fundus boundary — meaning the model's decision was driven predominantly "
                        f"by structures in that region rather than distributed evenly across the retina."
                    )

                    if trajectory_alert == "ACCELERATING":
                        xai_text += (
                            f" <span style='color:{DANGER}; font-weight:700;'>⚠ Longitudinal tracking across {len(record_logs)} visits "
                            f"indicates an upward pathology velocity (+{slope:.1f}% attention-index shift per visit, fit quality R²={r_squared:.2f}). "
                            f"This trend is flagged for clinician review — it is a statistical signal, not a diagnosis.</span>"
                        )
                    elif trajectory_alert == "NOISY_TREND":
                        xai_text += (
                            f" <span style='color:{WARN}; font-weight:600;'>A rising trend is present (+{slope:.1f}% per visit) but the "
                            f"fit is weak (R²={r_squared:.2f}), meaning the data points don't sit close to a clean line — more visits "
                            f"are needed before this trend can be treated as reliable.</span>"
                        )
                    elif trajectory_alert == "STABILIZED":
                        xai_text += " Longitudinal tracking across visits shows no statistically meaningful upward trend in attention intensity."
                    else:
                        xai_text += (
                            f" <span style='opacity:0.75;'>Longitudinal trend analysis is unavailable — at least 3 visits are required "
                            f"for a statistically defensible trend line ({len(record_logs)} on record for this patient).</span>"
                        )

                xai_text += (
                    " <span style='opacity:0.75; font-style:italic;'>Note: this explanation reflects where the model's attention "
                    "was concentrated, not a confirmed clinical diagnosis — attention correlates with, but does not prove, the "
                    "presence of a specific pathology at that location.</span>"
                )

                st.markdown(f"""
                <div class="rs-banner" style="background: {XAI_BG}; border-color: {XAI_BORDER};">
                    <div style="font-size: 22px; margin-top:-2px;">🧠</div>
                    <div>
                        <div class="rs-banner-title" style="color: {XAI_TEXT};">Explainable AI &amp; Predictive Prognostics</div>
                        <div class="rs-banner-body" style="color: #c7d6ff;"><strong>AI Decision Rationale:</strong> {xai_text}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="rs-banner" style="background: {WARN_BG}; border-color: {WARN_BORDER};">
                    <div style="font-size: 22px; margin-top:-2px;">📋</div>
                    <div>
                        <div class="rs-banner-title" style="color: {WARN};">Clinical Management Directive</div>
                        <div class="rs-banner-body" style="color: #f3ddaa;">{CLINICAL_DIRECTIVES[pred_idx]}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ===================== TAB 2: PROBABILITY & UNCERTAINTY =====================
            with tab_probability:
                st.markdown("<div class='rs-section-label'>Grade Probabilities Matrix</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='rs-card'>", unsafe_allow_html=True)
                for i in range(NUM_CLASSES):
                    cname = CLASS_NAMES[i]
                    pct = float(probabilities[i])
                    uncertainty = float(class_uncertainty[i])
                    is_pred = (i == pred_idx)
                    label_color = TEXT_MAIN if is_pred else TEXT_MUTED
                    weight = 700 if is_pred else 500
                    st.markdown(
                        f"<div style='font-size: 13px; color: {label_color}; display: flex; justify-content: space-between; font-weight:{weight}; margin-bottom:4px;'>"
                        f"<span>{cname}</span><span>{pct*100:.1f}% <span style='color:{TEXT_MUTED}; font-weight:400; font-size:11px;'>(± {uncertainty:.1f})</span></span></div>",
                        unsafe_allow_html=True
                    )
                    st.progress(pct)
                st.markdown(
                    f"<p style='color:{TEXT_MUTED}; font-size:11.5px; margin-top:12px; line-height:1.5;'>± values show predictive uncertainty (std. dev.) across the 3-view TTA ensemble — wider bands mean the augmented views disagreed more on that class.</p>",
                    unsafe_allow_html=True
                )
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("<br><div class='rs-section-label'>Referral Classification</div>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class="rs-card">
                    <p style="color:{TEXT_MUTED}; font-size:12.5px; line-height:1.7; margin:0;">
                        Standard DR screening convention groups the 5-class grade into a binary referral decision:
                        <b style="color:{TEXT_MAIN};">No DR / Mild NPDR</b> → non-referable (routine monitoring), and
                        <b style="color:{TEXT_MAIN};">Moderate NPDR and above</b> → referable (specialist review required).
                        This patient's aggregated referable-class probability is
                        <b style="color:{referral_color};">{referable_prob:.1f}%</b>.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # ===================== TAB 3: ADVANCED IMAGING =====================
            with tab_advanced:
                st.markdown("<div class='rs-section-label'>Vascular Topology &amp; AI ROI Focus</div>", unsafe_allow_html=True)
                adv_col1, adv_col2 = st.columns(2, gap="large")
                with adv_col1:
                    st.markdown("<span class='rs-subtle-label'>Vascular Topology Map</span>", unsafe_allow_html=True)
                    st.image(vessel_map, use_container_width=True, clamp=True)
                with adv_col2:
                    st.markdown(f"<span class='rs-subtle-label-accent'>AI ROI Bounding Boxes ({attention_index:.1f} Index)</span>", unsafe_allow_html=True)
                    st.image(boundary_rgb, use_container_width=True)

            # ===================== TAB 4: PATIENT HISTORY =====================
            with tab_history:
                hist_col1, hist_col2 = st.columns([1.3, 1], gap="large")
                with hist_col1:
                    st.markdown(f"<div class='rs-section-label'>Historical Tracking Summary Log — {patient_id}</div>", unsafe_allow_html=True)
                    table_html = '<table class="rs-table"><thead><tr>'
                    table_html += '<th>Visit Index</th>'
                    table_html += '<th>Timestamp</th>'
                    table_html += '<th>Diagnosis Verdict</th>'
                    table_html += '<th>Confidence Score</th>'
                    table_html += '<th>Attention Index</th>'
                    table_html += '</tr></thead><tbody>'
                    for i, r in enumerate(record_logs):
                        table_html += '<tr>'
                        table_html += f'<td><b>#{i+1}</b></td>'
                        table_html += f'<td>{r["timestamp"]}</td>'
                        table_html += f'<td>{r["diagnosis"]}</td>'
                        table_html += f'<td>{r["confidence"]}%</td>'
                        table_html += f'<td style="color:{ACCENT};"><b>{r["attention_index"]:.1f}%</b></td>'
                        table_html += '</tr>'
                    table_html += '</tbody></table>'
                    st.markdown(table_html, unsafe_allow_html=True)

                with hist_col2:
                    st.markdown("<div class='rs-section-label'>Diagnostics Forecast (Longitudinal)</div>", unsafe_allow_html=True)
                    fig, ax = plt.subplots(figsize=(4.2, 3.6))

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
                    ax.grid(True, linestyle='--', alpha=0.2, color=TEXT_MUTED)

                    fig.patch.set_facecolor(SURFACE)
                    ax.set_facecolor(SURFACE_ALT)
                    ax.spines['bottom'].set_color(BORDER)
                    ax.spines['left'].set_color(BORDER)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.tick_params(colors=TEXT_MUTED, labelsize=8)
                    st.pyplot(fig)
                    plt.close(fig)

            st.markdown("<br>", unsafe_allow_html=True)

            # --- DYNAMIC EXPORT BUTTON ASSEMBLIES ---
            ist_timezone = pytz.timezone('Asia/Kolkata')
            current_date_ist = datetime.now(ist_timezone).strftime('%Y%m%d')

            pdf_bytes = generate_clinical_pdf(
                patient_id, pred_name, confidence, attention_index,
                dominant_quad, quad_pct, CLINICAL_DIRECTIVES[pred_idx], consensus_status
            )

            st.download_button(
                label="📥  Export Signed Clinical Summary PDF",
                data=pdf_bytes,
                file_name=f"RetiScan_{patient_id}_{current_date_ist}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
    else:
        st.markdown(f"""
        <div class="rs-card" style="text-align:center; padding:70px 30px; margin-top:10px;">
            <div style="font-size:36px; margin-bottom:12px;">🖼️</div>
            <div style="color:{TEXT_MAIN}; font-size:16px; font-weight:700; margin-bottom:6px;">No retinal image loaded</div>
            <div style="color:{TEXT_MUTED}; font-size:13px;">Upload a fundus image from the Clinical Control Hub sidebar to begin analysis.</div>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("👈 Complete the entry inputs in the Clinical Control Hub sidebar to initialize data processing pathways.")

# =====================================================================
#  FOOTER — MODEL CARD & AUDIT TRAIL (compact, collapsed by default)
# =====================================================================
st.markdown("<div style='margin-top:38px;'></div>", unsafe_allow_html=True)
with st.expander("🗂️  Model Card & Audit Trail", expanded=False):
    mc_col1, mc_col2 = st.columns(2, gap="large")
    with mc_col1:
        st.markdown(f"""
        <div class="rs-footer-card">
            <p class="rs-subtle-label-accent" style="margin-bottom:10px;">Model Details</p>
            <p style="font-size:12.5px; color:{TEXT_MUTED}; line-height:1.85; margin:0;">
                <b style="color:{TEXT_MAIN};">Architecture:</b> {MODEL_CARD['architecture']}<br>
                <b style="color:{TEXT_MAIN};">Input:</b> {MODEL_CARD['input_resolution']}<br>
                <b style="color:{TEXT_MAIN};">Training Data:</b> {MODEL_CARD['training_dataset']}<br>
                <b style="color:{TEXT_MAIN};">Task:</b> {MODEL_CARD['num_classes']}<br>
                <b style="color:{TEXT_MAIN};">Loss:</b> {MODEL_CARD['loss_function']}<br>
                <b style="color:{TEXT_MAIN};">Reported Accuracy:</b> {MODEL_CARD['reported_accuracy']}<br>
                <b style="color:{TEXT_MAIN};">Explainability:</b> {MODEL_CARD['explainability_method']}
                {f" — active layer: <code>{gradcam_layer_name}</code>" if model_loaded else ""}<br>
                <b style="color:{TEXT_MAIN};">Uncertainty Estimate:</b> {MODEL_CARD['uncertainty_method']}
            </p>
        </div>
        """, unsafe_allow_html=True)
    with mc_col2:
        limitations_html = "".join([f"<li style='margin-bottom:6px;'>{item}</li>" for item in MODEL_CARD["known_limitations"]])
        st.markdown(f"""
        <div class="rs-footer-card">
            <p class="rs-subtle-label-accent" style="margin-bottom:10px;">Known Limitations</p>
            <ul style="font-size:12px; color:{TEXT_MUTED}; line-height:1.6; margin:0; padding-left:16px;">
                {limitations_html}
            </ul>
            <p class="rs-subtle-label-accent" style="margin: 14px 0 6px 0;">Intended Use</p>
            <p style="font-size:12px; color:{TEXT_MUTED}; line-height:1.6; margin:0;">{MODEL_CARD['intended_use']}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown(f"""
<div style="text-align:center; padding: 22px 0 6px 0; color:{TEXT_MUTED}; font-size:11px; letter-spacing:0.3px;">
    RetiScan Pro v5 &nbsp;·&nbsp; Decision-support tool for DR screening triage &nbsp;·&nbsp; Not a standalone diagnostic device
</div>
""", unsafe_allow_html=True)
