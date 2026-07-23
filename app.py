import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import sys
import gdown
import json
import hashlib
import matplotlib.pyplot as plt
import pytz
from datetime import datetime
from fpdf import FPDF
from skimage.filters import frangi  # Mathematical vascular extraction (DR-side vascular topology view only)
from skimage.morphology import skeletonize

# Set page configurations — wide canvas, no sidebar
st.set_page_config(
    page_title="RetiScan Pro v5",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================================================
#  0. DESIGN TOKENS
# =====================================================================
BG           = "#0d0e12"
SURFACE      = "#16171d"
SURFACE_ALT  = "#1d1f27"
RAISED       = "#24262f"
BORDER       = "#2c2e38"
TEXT_MAIN    = "#f3f2ee"
TEXT_MUTED   = "#96979f"
TEXT_FAINT   = "#5f606a"
ACCENT       = "#ff8a3d"
ACCENT_DEEP  = "#e56f22"
EMERALD      = "#33c793"
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

HR_SEVERITY_COLOR = {
    "Grade 0 (No HR)":       EMERALD,
    "Grade 1 (Mild HR)":     WARN,
    "Grade 2 (Moderate HR)": INFO,
    "Grade 3 (Severe HR)":   DANGER,
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

    /* ============ Radio Segmented Control ============ */
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

    /* ============ Misc ============ */
    hr {{ border-color: {BORDER} !important; }}
    ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
    ::-webkit-scrollbar-thumb {{ background-color: {BORDER}; border-radius: 8px; }}

    /* ============ Custom Layout Elements ============ */
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

    .rs-warn-box {{
        background: {SURFACE}; border-left: 4px solid {WARN}; border-radius: 10px;
        padding: 16px 20px; color: {TEXT_MAIN};
    }}

    .rs-timeline-chip {{
        display: inline-flex; flex-direction: column; gap: 2px;
        background: {SURFACE_ALT}; border: 1px solid {BORDER}; border-radius: 12px;
        padding: 10px 16px; margin-right: 10px; margin-bottom: 10px; min-width: 140px;
    }}
    </style>
""", unsafe_allow_html=True)

# =====================================================================
#  1. CONSTANTS, CLOUD PATHS, & METADATA
# =====================================================================
IMG_SIZE = 224
NUM_CLASSES = 5
MODEL_FILENAME = "retiscan_pro_v5_best.keras"
FILE_ID = "1NFcXDWOMIVyVbA9j2pXUR6b8kCYGVKyq"
HISTORY_FILE = "patient_history.json"
RRWNET_DIR = "rrwnet_lib"  # vendor model.py here to avoid runtime git-clone dependency

# --- Original DR Metadata (UNTOUCHED) ---
CLASS_NAMES = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR"
}

CLASS_DESCRIPTIONS = {
    0: "No visible signs of diabetic retinopathy. The retinal vasculature, macula, and optic disc region show no microaneurysms, "
       "hemorrhages, hard exudates, or neovascularization within the analyzed field.",
    1: "Mild Non-Proliferative DR — characterized by isolated microaneurysms: small, localized outpouchings in retinal capillary walls "
       "caused by chronic hyperglycemic damage. No exudates or gross hemorrhages are typically present yet.",
    2: "Moderate Non-Proliferative DR — progressive stage with multiple microaneurysms, intraretinal hemorrhages ('dot-and-blot' pattern), "
       "and early vascular abnormalities. Threshold for mandatory specialist referral.",
    3: "Severe Non-Proliferative DR — extensive hemorrhages across multiple quadrants, venous beading, or intraretinal microvascular "
       "abnormalities (IRMA). Elevated short-term risk of progressing to proliferative disease.",
    4: "Proliferative DR — advanced stage defined by neovascularization (fragile new blood vessel proliferation). Prone to leakage and rupture, "
       "creating immediate risk of vitreous hemorrhage or tractional retinal detachment."
}

CLINICAL_DIRECTIVES = {
    0: "Schedule routine tracking examination in 12 months. Continue baseline systemic blood glucose and blood pressure monitoring.",
    1: "Schedule a targeted Optical Coherence Tomography (OCT) review focusing on localized structural fields. Re-examine in 6-12 months.",
    2: "Refer to an ophthalmologist for urgent structural baseline tracking. Evaluate macular edema and stabilize HbA1c variables.",
    3: "Immediate specialist referral required. Initiate rapid panretinal photocoagulation (PRP) screening profiles safely.",
    4: "CRITICAL ALERT: Emergency vitreo-retinal surgical evaluation indicated. High immediate risk of permanent tractional detachment."
}

REFERABLE_CLASSES = {2, 3, 4}

# --- HR Metadata (Keith-Wagener-Barker Scale, Grades 0-3 reachable via AVR) ---
# Grade 4 (malignant HR / papilledema) is NOT derivable from AVR alone and is
# explicitly out of scope until a disc-swelling detector is added. See Model Card.
HR_CLASS_NAMES = {
    0: "Grade 0 (No HR)",
    1: "Grade 1 (Mild HR)",
    2: "Grade 2 (Moderate HR)",
    3: "Grade 3 (Severe HR)",
}

HR_CLASS_DESCRIPTIONS = {
    0: "Arteriolar-to-venular ratio within normal range. No evidence of generalized or focal arteriolar narrowing at the measured B-zone.",
    1: "Mild generalized retinal arteriolar narrowing (reduced AVR). Early vascular response to systemic hypertension.",
    2: "Marked arteriolar attenuation consistent with focal narrowing and/or AV nicking at vascular crossings.",
    3: "Severely reduced AVR consistent with significant microvascular injury. Correlate clinically for hemorrhages, cotton-wool spots, and hard exudates.",
}

HR_CLINICAL_DIRECTIVES = {
    0: "Maintain routine annual cardiovascular and fundus tracking. Continue standard lifestyle and blood pressure targets (<130/80 mmHg).",
    1: "Advise primary care physician (PCP) for baseline 24-hour ambulatory blood pressure monitoring (ABPM). Re-evaluate fundus in 12 months.",
    2: "Refer to primary care/cardiology for optimized antihypertensive regimen adjustment. Recheck retinal microvasculature in 3-6 months.",
    3: "URGENT SYSTEMIC EVALUATION REQUIRED: Contact managing physician within 24-48 hours. Target gradual blood pressure reduction.",
}

MODEL_CARD = {
    "architecture": "DR: EfficientNetB3 (ImageNet backbone). HR: RRWNet (pretrained artery/vein segmentation, Morano et al. 2024) + clinical Knudtson-Parr-Hubbard AVR computation.",
    "input_resolution": f"DR: {IMG_SIZE} x {IMG_SIZE} RGB. HR: native resolution, downscaled to max 768px for inference.",
    "training_dataset": "DR: APTOS 2019 Blindness Detection dataset. HR: RRWNet pretrained on RITE/LES-AV/HRF (not retrained by this project).",
    "num_classes": "DR: 5-class ICDR. HR: Grades 0-3 of the Keith-Wagener-Barker scale (Grade 4 out of scope, see limitations).",
    "loss_function": "DR: Categorical Crossentropy / Focal Loss.",
    "reported_accuracy": "DR: ~89% top-1 accuracy on validation split. HR: inherits RRWNet's published segmentation benchmarks; AVR-to-grade mapping is rule-based and not independently validated by this project on labeled KWB data yet.",
    "explainability_method": "DR: Grad-CAM, quadrant-mapped. HR: visual artery/vein segmentation masks + numeric AVR/CRAE/CRVE, computed via the clinically standard B-zone + Knudtson formula.",
    "uncertainty_method": "DR: Test-Time Augmentation (TTA) ensemble variance across 3 views. HR: explicit 'indeterminate' abstain state when fewer than 2 reliable vessels are resolved in the B-zone — no forced grade on weak signal.",
    "known_limitations": [
        "HR Grade 4 (malignant HR / papilledema) is not derivable from AVR alone and is currently unreachable — flagged, not silently mis-graded.",
        "HR inference uses simplified preprocessing (resize + normalize) rather than RRWNet's original CLAHE-based pipeline, which may reduce accuracy below the paper's reported benchmarks.",
        "The AVR-to-KWB-grade cutoffs used here are literature-informed thresholds, not independently calibrated against a labeled KWB dataset by this project.",
        "DR predictions rely strictly on original EfficientNet preprocessing to maintain baseline accuracy.",
        "Not a standalone diagnostic tool. Intended as a dual-screening decision-support triage aid."
    ],
    "intended_use": "Screening triage support for combined diabetic and hypertensive retinopathy grading.",
}

def focal_loss():
    def loss_fn(y_true, y_pred): return tf.reduce_mean(y_pred)
    loss_fn.__name__ = "focal_loss"
    return loss_fn

# =====================================================================
#  2. CACHED DR MODEL ENGINE & LONGITUDINAL DATABASE (UNTOUCHED)
# =====================================================================
@st.cache_resource
def load_retiscan_pipeline():
    if not os.path.exists(MODEL_FILENAME):
        with st.spinner("Downloading trained model weights from cloud storage..."):
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
    st.error(f"Could not initialize DR model. Error: {e}")
    model_loaded = False

def load_patient_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f: return json.load(f)
        except: return {}
    return {}

def save_patient_record(p_id, diagnosis, confidence, attention_index, hr_diagnosis="N/A"):
    history = load_patient_history()
    if p_id not in history: history[p_id] = []

    ist_timezone = pytz.timezone('Asia/Kolkata')
    current_time_ist = datetime.now(ist_timezone).strftime("%Y-%m-%d %H:%M")

    new_entry = {
        "timestamp": current_time_ist,
        "diagnosis": diagnosis,
        "hr_diagnosis": hr_diagnosis,
        "confidence": round(float(confidence), 2),
        "attention_index": round(float(attention_index), 2)
    }
    history[p_id].append(new_entry)
    with open(HISTORY_FILE, "w") as f: json.dump(history, f, indent=4)
    return history[p_id]

# =====================================================================
#  3. ORIGINAL UNTOUCHED DR PREPROCESSING & INFERENCE ENGINE
# =====================================================================
def preprocess_for_inference(img_bgr):
    """ORIGINAL UNTOUCHED DR PREPROCESSING. EfficientNet handles internal rescaling."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
    tensor = rgb.astype(np.float32)
    return np.expand_dims(tensor, axis=0)

def run_tta_ensemble_inference(model_obj, base_tensor):
    """ORIGINAL TTA ENSEMBLE FOR DR."""
    pred_base = model_obj.predict_on_batch(base_tensor)[0]

    flipped_tensor = np.flip(base_tensor, axis=2)
    pred_flipped = model_obj.predict_on_batch(flipped_tensor)[0]

    gamma_tensor = np.clip(base_tensor * 1.05, 0.0, 255.0)
    pred_gamma = model_obj.predict_on_batch(gamma_tensor)[0]

    fused_probabilities = (pred_base + pred_flipped + pred_gamma) / 3.0

    per_class_variance = np.var([pred_base, pred_flipped, pred_gamma], axis=0)
    mean_variance = np.mean(per_class_variance)
    consensus_badge = "HIGH CONSENSUS" if mean_variance < 0.02 else "BORDERLINE VERIFICATION REQUIRED"
    per_class_uncertainty_pct = np.sqrt(per_class_variance) * 100.0

    return fused_probabilities, consensus_badge, per_class_uncertainty_pct

# =====================================================================
#  4. HYPERTENSIVE RETINOPATHY (HR) ENGINE — fully isolated from DR
#  Public entry point takes ONLY the raw image + fundus geometry.
#  Never reads dr_pred_idx, DR probabilities, or any DR-pipeline state.
# =====================================================================
@st.cache_resource(show_spinner="Loading HR vascular model (one-time)...")
def load_hr_model():
    """
    Loads pretrained RRWNet A/V segmentation weights (Hugging Face).
    Cached once per server process. torch is imported lazily so app
    startup stays light until an image is actually uploaded.
    """
    import torch

    if not os.path.exists(RRWNET_DIR):
        exit_code = os.system(f"git clone --depth 1 https://github.com/j-morano/rrwnet {RRWNET_DIR} 2>/dev/null")
        if exit_code != 0 or not os.path.exists(RRWNET_DIR):
            raise RuntimeError(
                "Could not fetch RRWNet architecture definition. "
                "Recommended fix: vendor rrwnet_lib/model.py directly into the repo "
                "instead of relying on a runtime git clone."
            )

    if RRWNET_DIR not in sys.path:
        sys.path.insert(0, RRWNET_DIR)

    from model import RRWNet as RRWNetModel
    from huggingface_hub import PyTorchModelHubMixin

    class RRWNet(RRWNetModel, PyTorchModelHubMixin):
        def __init__(self, input_ch=3, output_ch=3, base_ch=64, iterations=5):
            super().__init__(input_ch, output_ch, base_ch, iterations)

    # Cap CPU threads so PyTorch doesn't starve TensorFlow's DR inference
    # running in the same process.
    torch.set_num_threads(2)

    hr_model = RRWNet.from_pretrained("j-morano/rrwnet-rite")
    hr_model.eval()
    return hr_model


def run_av_segmentation(img_bgr, max_dim=768):
    """
    Runs A/V segmentation. Image is downscaled before inference — this is
    the main lever against OOM crashes on memory-constrained deployments.
    """
    import torch

    hr_model = load_hr_model()
    h0, w0 = img_bgr.shape[:2]
    scale = min(1.0, max_dim / max(h0, w0))
    img_small = cv2.resize(img_bgr, (int(w0 * scale), int(h0 * scale))) if scale < 1.0 else img_bgr.copy()

    rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB).astype(np.float32)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0) / 255.0

    with torch.no_grad():
        output = hr_model(tensor)

    pred = output[-1] if isinstance(output, (list, tuple)) else output  # final refined iteration
    pred = torch.sigmoid(pred)[0].permute(1, 2, 0).numpy()
    pred_full = cv2.resize(pred, (w0, h0))

    artery_mask = (pred_full[..., 0] > 0.5).astype(np.uint8) * 255
    vein_mask   = (pred_full[..., 1] > 0.5).astype(np.uint8) * 255
    vessel_mask = (pred_full[..., 2] > 0.5).astype(np.uint8) * 255
    return artery_mask, vein_mask, vessel_mask


def detect_optic_disc(img_bgr, fundus_x, fundus_y, fundus_radius):
    """Brightness-based disc localization within the known fundus circle."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    circle_mask = np.zeros_like(gray)
    cv2.circle(circle_mask, (int(fundus_x), int(fundus_y)), int(fundus_radius * 0.9), 255, -1)
    masked = cv2.bitwise_and(gray, gray, mask=circle_mask)
    blurred = cv2.GaussianBlur(masked, (25, 25), 0)
    _, _, _, max_loc = cv2.minMaxLoc(blurred)
    disc_radius = fundus_radius * 0.12  # typical disc:fundus ratio
    return max_loc[0], max_loc[1], disc_radius


def compute_b_zone_mask(h, w, disc_x, disc_y, disc_radius):
    """Standard clinical B-zone: 0.5-1.0 disc diameters from the disc margin."""
    inner_r, outer_r = disc_radius * 1.5, disc_radius * 2.0
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - disc_x) ** 2 + (yy - disc_y) ** 2)
    return (((dist >= inner_r) & (dist <= outer_r)).astype(np.uint8)) * 255


def measure_vessel_calibers(vessel_channel_mask, b_zone_mask, top_n=6):
    """Approximates vessel width per segment as area / skeleton-length within the B-zone."""
    masked = cv2.bitwise_and(vessel_channel_mask, b_zone_mask)
    if cv2.countNonZero(masked) < 20:
        return []
    skeleton = skeletonize(masked > 0)
    num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats((masked > 0).astype(np.uint8), connectivity=8)
    widths = []
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        skel_len = np.sum(skeleton & (labels_im == lbl))
        if skel_len > 3:
            widths.append(area / skel_len)
    return sorted(widths, reverse=True)[:top_n]


def knudtson_avr(artery_widths, vein_widths):
    """Knudtson-Parr-Hubbard iterative pairing formula -> CRAE, CRVE, AVR."""
    def combine(widths, is_artery):
        w = sorted(widths)
        while len(w) > 1:
            new_w, i, j = [], 0, len(w) - 1
            while i < j:
                coef = 0.88 if is_artery else 0.95
                new_w.append(coef * np.sqrt(w[i] ** 2 + w[j] ** 2))
                i += 1; j -= 1
            if i == j:
                new_w.append(w[i])
            w = sorted(new_w)
        return w[0] if w else None

    if len(artery_widths) < 2 or len(vein_widths) < 2:
        return None, None, None  # not enough resolved vessels -> abstain, don't force a grade
    crae, crve = combine(artery_widths, True), combine(vein_widths, False)
    if not crae or not crve or crve == 0:
        return None, None, None
    return crae, crve, crae / crve


def analyze_hypertensive_retinopathy(img_bgr, x_center, y_center, radius):
    """
    PUBLIC ENTRY POINT. Signature takes ONLY raw image + fundus geometry —
    no DR prediction, no DR probabilities, no shared state with the DR path.
    Fails soft: any error returns an 'error'/'indeterminate' status instead
    of raising and crashing the Streamlit process.
    """
    try:
        artery_mask, vein_mask, vessel_mask = run_av_segmentation(img_bgr)
    except Exception as e:
        return {"status": "error", "pred_name": "HR Engine Unavailable",
                "message": f"Segmentation failed safely: {e}"}

    try:
        disc_x, disc_y, disc_r = detect_optic_disc(img_bgr, x_center, y_center, radius)
        h, w = img_bgr.shape[:2]
        b_zone = compute_b_zone_mask(h, w, disc_x, disc_y, disc_r)

        artery_widths = measure_vessel_calibers(artery_mask, b_zone)
        vein_widths = measure_vessel_calibers(vein_mask, b_zone)
        crae, crve, avr = knudtson_avr(artery_widths, vein_widths)
    except Exception as e:
        return {"status": "error", "pred_name": "HR Analysis Failed",
                "message": f"AVR computation failed safely: {e}",
                "artery_mask": artery_mask, "vein_mask": vein_mask, "vessel_mask": vessel_mask}

    if avr is None:
        return {"status": "indeterminate", "pred_name": "Indeterminate — Insufficient Vascular Signal",
                "message": "Not enough clearly resolved arteries/veins in the B-zone for a reliable AVR.",
                "artery_mask": artery_mask, "vein_mask": vein_mask, "vessel_mask": vessel_mask}

    if avr >= 0.65:
        idx, name = 0, "Grade 0 (No HR)"
    elif avr >= 0.58:
        idx, name = 1, "Grade 1 (Mild HR)"
    elif avr >= 0.50:
        idx, name = 2, "Grade 2 (Moderate HR)"
    else:
        idx, name = 3, "Grade 3 (Severe HR)"
    # Grade 4 (malignant) requires papilledema detection, not derivable from
    # AVR alone — explicit scope limitation until a disc-swelling detector
    # is added (see Model Card).

    return {"status": "ok", "pred_idx": idx, "pred_name": name, "avr": round(float(avr), 3),
            "crae": round(float(crae), 2), "crve": round(float(crve), 2),
            "artery_mask": artery_mask, "vein_mask": vein_mask, "vessel_mask": vessel_mask}

# =====================================================================
#  5. AUXILIARY SCREENING & REPORT GENERATION UTILITIES
# =====================================================================
def generate_clinical_pdf(p_id, verdict, conf, attn_idx, quad, quad_pct, directive, consensus, hr_results):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(16, 185, 129)
    pdf.cell(0, 10, "RETISCAN PRO DIAGNOSTIC SUMMARY REPORT", ln=True, align="C")
    pdf.ln(6)

    ist_timezone = pytz.timezone('Asia/Kolkata')
    current_time_ist = datetime.now(ist_timezone).strftime("%Y-%m-%d %H:%M")

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 7, f"Patient Tracking Key: {p_id}", ln=True)
    pdf.cell(0, 7, f"Generated Timestamp: {current_time_ist}", ln=True)
    pdf.cell(0, 7, f"Ensemble Integrity State: {consensus}", ln=True)
    pdf.line(10, pdf.get_y() + 2, 200, pdf.get_y() + 2)
    pdf.ln(6)

    # DR Section
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(229, 111, 34)
    pdf.cell(0, 8, f"1. Diabetic Retinopathy (DR): {verdict}", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 7, f"   - DR Confidence: {conf:.2f}%", ln=True)
    pdf.cell(0, 7, f"   - Neuro-Attention Mapping Index: {attn_idx:.1f}%", ln=True)
    pdf.cell(0, 7, f"   - Dominant Focus: {quad} Quadrant ({quad_pct:.1f}%)", ln=True)
    pdf.ln(3)

    # HR Section
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(94, 177, 239)
    pdf.cell(0, 8, f"2. Hypertensive Retinopathy (HR): {hr_results.get('pred_name', 'N/A')}", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(0, 0, 0)
    if hr_results.get("status") == "ok":
        pdf.cell(0, 7, f"   - Arteriolar-to-Venular Ratio (AVR): {hr_results['avr']}", ln=True)
        pdf.cell(0, 7, f"   - CRAE: {hr_results['crae']}  |  CRVE: {hr_results['crve']}", ln=True)
        pdf.cell(0, 7, f"   - Scale: Keith-Wagener-Barker Grades 0-3 (Grade 4 out of current scope)", ln=True)
    else:
        pdf.cell(0, 7, f"   - Status: {hr_results.get('status', 'unknown').upper()}", ln=True)
        pdf.multi_cell(0, 6, f"   - {hr_results.get('message', '')}")
    pdf.ln(6)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Official Management Protocol Directives:", ln=True)
    pdf.set_font("Helvetica", "I", 10)
    hr_directive = HR_CLINICAL_DIRECTIVES.get(hr_results.get("pred_idx"), "N/A — see HR status above.")
    pdf.multi_cell(0, 5, f"DR: {directive}\nHR: {hr_directive}")

    return bytes(pdf.output())

def run_pre_computing_screening(img_bgr):
    h, w, _ = img_bgr.shape
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < 25.0:
        return False, f"REJECTED: Image fails focal clarity standards (Blur Variance: {blur_score:.1f}). Please recapture.", None, None, None

    mean_brightness = np.mean(gray)
    if mean_brightness < 8:
        return False, "REJECTED: Low image luminance exposure (Image too dark).", None, None, None

    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return False, "REJECTED: Geometry unverified. No structural contour field found.", None, None, None

    largest_contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(largest_contour)
    total_area = h * w

    if contour_area < (total_area * 0.12):
        return False, "REJECTED: Fundus structure area is too small relative to frame.", None, None, None

    (x_center, y_center), radius = cv2.minEnclosingCircle(largest_contour)
    circle_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(circle_mask, (int(x_center), int(y_center)), int(radius * 0.8), 255, -1)

    mean_b, mean_g, mean_r, _ = cv2.mean(img_bgr, mask=circle_mask)
    if mean_b > 115:
        return False, f"REJECTED: Biological profile mismatch (Anomalous blue channel reflection detected).", None, None, None

    return True, "PASSED", x_center, y_center, radius

def generate_vascular_map(img_bgr, x_center, y_center, radius):
    """DR-side vascular topology view only — independent of the HR engine."""
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

def compute_diagnostic_graphs(img_tensor, grad_model_obj, pred_idx, img_bgr, x_center, y_center, radius):
    h, w, _ = img_bgr.shape

    with tf.GradientTape() as tape:
        conv_outputs, model_predictions = grad_model_obj(img_tensor)
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
#  6. USER INTERFACE
# =====================================================================

# HERO HEADER
st.markdown(f"""
<div class="rs-hero">
    <div class="rs-hero-mark">🩺</div>
    <div>
        <div class="rs-hero-title">RetiScan Pro <span style="color:{ACCENT};">v5</span></div>
        <div class="rs-hero-sub">AI-assisted Diabetic & Hypertensive Retinopathy multi-diagnostic grading</div>
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
    st.markdown('<div class="rs-toolbar-label">Active Pipelines</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:13px; font-weight:700; color:{TEXT_MAIN}; margin-top:6px;">1. DR: EfficientNetB3 (ICDR)</div>
    <div style="font-size:11.5px; color:{TEXT_MUTED}; margin-top:2px;">2. HR: RRWNet A/V Segmentation + Knudtson AVR</div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

if not model_loaded:
    st.stop()

if uploaded_file is None:
    st.markdown(f"""
    <div style="text-align:center; padding:80px 20px; color:{TEXT_MUTED};">
        <div style="font-size:38px; margin-bottom:14px;">🖼️</div>
        <div style="color:{TEXT_MAIN}; font-size:16px; font-weight:700; margin-bottom:6px;">Awaiting retinal image</div>
        <div style="font-size:13px;">Upload a fundus photograph above to initiate multi-disease diagnostic screening.</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

# SESSION CACHE
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
            <span style='font-size:12px; opacity:0.75;'>Pipeline terminated automatically to prevent false classification predictions.</span>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # Preprocess DR tensor (0..255 range float32)
    img_tensor = preprocess_for_inference(img_bgr)

    with st.spinner("Executing DR Ensemble Analytics..."):
        probabilities, consensus_status, class_uncertainty = run_tta_ensemble_inference(model, img_tensor)

    pred_idx = int(np.argmax(probabilities))
    pred_name = CLASS_NAMES[pred_idx]
    confidence = probabilities[pred_idx] * 100

    isolated_heatmap, boundary_img, attention_index, dominant_quad, quad_pct = compute_diagnostic_graphs(
        img_tensor, grad_model, pred_idx, img_bgr, x_center, y_center, radius
    )

    with st.spinner("Analyzing Hypertensive Retinopathy (isolated A/V pipeline)..."):
        vessel_map = generate_vascular_map(img_bgr, x_center, y_center, radius)  # DR display only
        # NOTE: HR call takes ONLY the raw image + fundus geometry — no dr_pred_idx,
        # no DR probabilities, no vessel_map. Independence is structural, not by convention.
        hr_results = analyze_hypertensive_retinopathy(img_bgr, x_center, y_center, radius)

    record_logs = save_patient_record(
        patient_id, pred_name, confidence, attention_index, hr_results.get("pred_name", "N/A")
    )

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
        "hr_results": hr_results,
        "record_logs": record_logs,
        "img_rgb": img_rgb,
        "gradcam_rgb": gradcam_rgb,
        "boundary_rgb": boundary_rgb,
        "vessel_map": vessel_map,
    }

# Retrieve cached values
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
hr_results         = cached["hr_results"]
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

# =====================================================================
#  SECTION 1: UNTOUCHED DIABETIC RETINOPATHY (DR) EVALUATION DECK
# =====================================================================
st.markdown('<div class="rs-divider-label">Diabetic Retinopathy (DR) Analysis Deck</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="rs-verdict-hero" style="background: linear-gradient(120deg, {accent_color}18, {SURFACE} 55%);">
    <div style="color:{TEXT_MUTED}; font-size:11px; text-transform:uppercase; letter-spacing:1px; font-weight:800;">Diabetic Diagnostic Verdict</div>
    <div style="font-size:42px; font-weight:800; color:{accent_color}; margin:6px 0 12px 0; letter-spacing:-0.5px;">{pred_name}</div>
    <div>
        <span class="rs-pill" style="border-color:{referral_color}; color:{referral_color}; background:{referral_color}14;">
            {"⬤" if is_referable else "○"} {referral_label}
        </span>
        <span class="rs-pill" style="border-color:{BORDER}; color:{TEXT_MUTED};">{consensus_status}</span>
    </div>
    <div class="rs-stat-strip">
        <div class="rs-stat"><div class="rs-stat-label">DR Confidence</div><div class="rs-stat-value">{confidence:.1f}%</div></div>
        <div class="rs-stat"><div class="rs-stat-label">Attention Intensity</div><div class="rs-stat-value">{attention_index:.1f}%</div></div>
        <div class="rs-stat"><div class="rs-stat-label">Referable Probability</div><div class="rs-stat-value" style="color:{referral_color};">{referable_prob:.1f}%</div></div>
        <div class="rs-stat"><div class="rs-stat-label">Dominant Quadrant</div><div class="rs-stat-value" style="font-size:15px;">{dominant_quad}</div></div>
    </div>
</div>
<p style="color:{TEXT_MAIN}; font-size:13.5px; line-height:1.65; margin: 14px 4px 0 4px;">{CLASS_DESCRIPTIONS[pred_idx]}</p>
""", unsafe_allow_html=True)

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
    visit_stamps = [r["timestamp"].split(" ")[0] for r in record_logs]
    attention_indices = [r["attention_index"] for r in record_logs]
    x_indices = np.arange(len(record_logs))

    MIN_VISITS_FOR_TREND = 3
    r_squared, slope, next_x, next_y_pred = None, None, None, None
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
        xai_text = "Grad-CAM attention shows diffuse, low-magnitude activation across the retina, consistent with a 'No DR' state."
    else:
        xai_text = (
            f"Grad-CAM traces primary diagnostic evidence to the <strong>{dominant_quad} quadrant</strong> "
            f"({quad_pct:.1f}% focus). "
        )
        if trajectory_alert == "ACCELERATING":
            xai_text += f"<span style='color:{DANGER}; font-weight:700;'>⚠ Upward pathology velocity (+{slope:.1f}%/visit) detected.</span>"

    st.markdown(f"""
    <div class="rs-rail-accent">
        <div class="rs-rail-title" style="color:{ACCENT};">DR Explainable AI Rationale</div>
        <div class="rs-rail-body">{xai_text}</div>
    </div>
    <div class="rs-rail-warn">
        <div class="rs-rail-title" style="color:{WARN};">DR Management Directive</div>
        <div class="rs-rail-body">{CLINICAL_DIRECTIVES[pred_idx]}</div>
    </div>
    """, unsafe_allow_html=True)

# =====================================================================
#  SECTION 2: HYPERTENSIVE RETINOPATHY (HR) EVALUATION DECK
#  Fully isolated pipeline — RRWNet A/V segmentation + Knudtson AVR.
# =====================================================================
st.markdown('<div class="rs-divider-label">Hypertensive Retinopathy (HR) Analysis Deck</div>', unsafe_allow_html=True)

if hr_results.get("status") != "ok":
    # Soft-fail states — engine error or insufficient vascular signal.
    # Never crash the page; always give the user a clear, honest status.
    icon = "⚠" if hr_results.get("status") == "indeterminate" else "✕"
    st.markdown(f"""
    <div class="rs-warn-box">
        <b style="color:{WARN};">{icon} {hr_results.get('pred_name', 'HR Unavailable')}</b><br><br>
        {hr_results.get('message', 'No further detail available.')}
        <div style="margin-top:10px; font-size:12px; color:{TEXT_MUTED};">
            This does not affect the Diabetic Retinopathy result above — the two pipelines are fully independent.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Still show the raw A/V masks if segmentation itself succeeded, even if AVR couldn't be computed
    if "artery_mask" in hr_results:
        m1, m2, m3 = st.columns(3)
        with m1: st.image(hr_results["artery_mask"], caption="Detected Arteries", use_container_width=True, clamp=True)
        with m2: st.image(hr_results["vein_mask"], caption="Detected Veins", use_container_width=True, clamp=True)
        with m3: st.image(hr_results["vessel_mask"], caption="Vessel Union", use_container_width=True, clamp=True)
else:
    hr_color = HR_SEVERITY_COLOR[hr_results["pred_name"]]
    hr_is_severe = hr_results["pred_idx"] >= 2
    hr_ref_color = DANGER if hr_is_severe else EMERALD

    hr_col1, hr_col2 = st.columns([1.35, 1], gap="large")

    with hr_col1:
        st.markdown(f"""
        <div class="rs-verdict-hero" style="background: linear-gradient(120deg, {hr_color}18, {SURFACE} 55%); border: 1px solid {BORDER}; padding: 24px 28px;">
            <div style="color:{TEXT_MUTED}; font-size:11px; text-transform:uppercase; letter-spacing:1px; font-weight:800;">Hypertensive Diagnostic Verdict</div>
            <div style="font-size:36px; font-weight:800; color:{hr_color}; margin:6px 0 10px 0; letter-spacing:-0.5px;">{hr_results['pred_name']}</div>
            <div>
                <span class="rs-pill" style="border-color:{hr_ref_color}; color:{hr_ref_color}; background:{hr_ref_color}14;">
                    {"⬤ CARDIO-VASCULAR ALERT" if hr_is_severe else "○ SYSTEMICALLY STABLE"}
                </span>
                <span class="rs-pill" style="border-color:{BORDER}; color:{TEXT_MUTED};">KWB SCALE (Grades 0-3)</span>
            </div>
            <div class="rs-stat-strip">
                <div class="rs-stat"><div class="rs-stat-label">AVR</div><div class="rs-stat-value">{hr_results['avr']}</div></div>
                <div class="rs-stat"><div class="rs-stat-label">CRAE</div><div class="rs-stat-value">{hr_results['crae']}</div></div>
                <div class="rs-stat"><div class="rs-stat-label">CRVE</div><div class="rs-stat-value">{hr_results['crve']}</div></div>
            </div>
        </div>
        <p style="color:{TEXT_MAIN}; font-size:13px; line-height:1.6; margin: 12px 4px 18px 4px;">{HR_CLASS_DESCRIPTIONS[hr_results['pred_idx']]}</p>
        """, unsafe_allow_html=True)

        st.markdown('<div class="rs-divider-label">A/V Segmentation Evidence</div>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        with m1: st.image(hr_results["artery_mask"], caption="Arteries", use_container_width=True, clamp=True)
        with m2: st.image(hr_results["vein_mask"], caption="Veins", use_container_width=True, clamp=True)
        with m3: st.image(hr_results["vessel_mask"], caption="Vessel Union", use_container_width=True, clamp=True)

    with hr_col2:
        st.markdown(f"""
        <div class="rs-rail-accent" style="border-left-color: {INFO};">
            <div class="rs-rail-title" style="color:{INFO};">Vascular Topography Rationale</div>
            <div class="rs-rail-body">
                Arteriolar-to-Venular Ratio (AVR) of <code>{hr_results['avr']}</code> computed via the clinically
                standard Knudtson-Parr-Hubbard formula, using vessel calibers measured in the B-zone
                (0.5-1.0 optic disc diameters from the disc margin) of independently segmented arteries
                (CRAE {hr_results['crae']}) and veins (CRVE {hr_results['crve']}).
            </div>
        </div>
        <div class="rs-rail-warn">
            <div class="rs-rail-title" style="color:{WARN};">HR Systemic Management Directive</div>
            <div class="rs-rail-body">{HR_CLINICAL_DIRECTIVES[hr_results['pred_idx']]}</div>
        </div>
        <div class="rs-rail">
            <div class="rs-rail-title" style="color:{TEXT_MUTED};">Multi-Disease Interaction Note</div>
            <div class="rs-rail-body" style="color:{TEXT_MUTED}; font-size:12.5px;">
                Concurrent Diabetic Retinopathy (<strong>{pred_name}</strong>) and Hypertensive Retinopathy (<strong>{hr_results['pred_name']}</strong>)
                exponentially increase the risk of macular edema and vision loss.
            </div>
        </div>
        """, unsafe_allow_html=True)

# =====================================================================
#  SECTION 3: PATIENT HISTORY & TREND GRAPH
# =====================================================================
st.markdown(f'<div class="rs-divider-label">Visit History & Timeline · {patient_id}</div>', unsafe_allow_html=True)

hist_col, chart_col = st.columns([1.2, 1], gap="large")

with hist_col:
    chips_html = ""
    for i, r in enumerate(record_logs):
        sev_color = SEVERITY_COLOR.get(r["diagnosis"], TEXT_MUTED)
        hr_diag = r.get("hr_diagnosis", "N/A")
        chips_html += f"""
        <div class="rs-timeline-chip">
            <span style="color:{TEXT_FAINT}; font-size:10px; text-transform:uppercase; font-weight:700;">Visit #{i+1} · {r['timestamp']}</span>
            <span style="color:{sev_color}; font-size:13.5px; font-weight:800;">DR: {r['diagnosis']}</span>
            <span style="color:{INFO}; font-size:11.5px; font-weight:700;">HR: {hr_diag}</span>
            <span style="color:{TEXT_MUTED}; font-size:10.5px;">Conf {r['confidence']}% · Attn {r['attention_index']:.1f}%</span>
        </div>
        """
    st.markdown(chips_html, unsafe_allow_html=True)

with chart_col:
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    ax.plot(x_indices, attention_indices, marker='o', color=ACCENT, linewidth=2.5, label='DR Attention')
    if len(record_logs) >= MIN_VISITS_FOR_TREND:
        forecast_x = [x_indices[-1], next_x]
        forecast_y = [attention_indices[-1], next_y_pred]
        ax.plot(forecast_x, forecast_y, linestyle='--', color=DANGER, linewidth=2, marker='x', label='DR Forecast')
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

# =====================================================================
#  SECTION 4: EXPORT SUMMARY PDF
# =====================================================================
ist_timezone = pytz.timezone('Asia/Kolkata')
current_date_ist = datetime.now(ist_timezone).strftime('%Y%m%d')

pdf_bytes = generate_clinical_pdf(
    patient_id, pred_name, confidence, attention_index,
    dominant_quad, quad_pct, CLINICAL_DIRECTIVES[pred_idx], consensus_status,
    hr_results
)

st.markdown("<br>", unsafe_allow_html=True)
st.download_button(
    label="📥  Export Multi-Diagnostic Summary PDF (DR + HR)",
    data=pdf_bytes,
    file_name=f"RetiScan_{patient_id}_{current_date_ist}.pdf",
    mime="application/pdf",
    use_container_width=True
)

# =====================================================================
#  SECTION 5: AUDIT TRAIL & MODEL CARD
# =====================================================================
st.markdown("<div style='margin-top:30px;'></div>", unsafe_allow_html=True)
with st.expander("🗂️  Model Card & Audit Trail", expanded=False):
    mc_col1, mc_col2 = st.columns(2, gap="large")
    with mc_col1:
        st.markdown(f"""
        <p style="font-size:12.5px; color:{TEXT_MUTED}; line-height:1.9; margin:0;">
            <b style="color:{TEXT_MAIN};">Architectures:</b> {MODEL_CARD['architecture']}<br>
            <b style="color:{TEXT_MAIN};">Input Standard:</b> {MODEL_CARD['input_resolution']}<br>
            <b style="color:{TEXT_MAIN};">Training Corpora:</b> {MODEL_CARD['training_dataset']}<br>
            <b style="color:{TEXT_MAIN};">Class Scope:</b> {MODEL_CARD['num_classes']}<br>
            <b style="color:{TEXT_MAIN};">Loss Function:</b> {MODEL_CARD['loss_function']}<br>
            <b style="color:{TEXT_MAIN};">Reported Accuracy:</b> {MODEL_CARD['reported_accuracy']}<br>
            <b style="color:{TEXT_MAIN};">Explainability Engine:</b> {MODEL_CARD['explainability_method']}<br>
            <b style="color:{TEXT_MAIN};">Uncertainty Pipeline:</b> {MODEL_CARD['uncertainty_method']}
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
    RetiScan Pro v5 &nbsp;·&nbsp; Dual DR & HR screening triage decision-support engine &nbsp;·&nbsp; Session: {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%d %b %Y, %H:%M IST')}
</div>
""", unsafe_allow_html=True)
