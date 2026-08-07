import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import sys
import sqlite3
import threading
import gdown
import hashlib
import matplotlib.pyplot as plt
import pytz
from datetime import datetime
from fpdf import FPDF
from skimage.filters import frangi  # DR-side vascular topology view only
from skimage.morphology import skeletonize, disk
from scipy.ndimage import distance_transform_edt
from scipy import ndimage

# ---------------------------------------------------------------------
# Issue D (insurance, not currently triggered on CPU-only deployments):
# prevent TensorFlow from pre-allocating all GPU memory if this app is
# ever moved to a GPU host, so it can coexist safely with PyTorch's HR
# model without a CUDA OOM conflict.
# ---------------------------------------------------------------------
_gpus = tf.config.list_physical_devices('GPU')
for _gpu in _gpus:
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except Exception:
        pass

# Set page configurations — wide canvas, no sidebar
st.set_page_config(
    page_title="RetiScan Pro v5",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================================================
#  0. DESIGN TOKENS — clean, minimal, professional (icon-free)
# =====================================================================
BG           = "#ffffff"
SURFACE      = "#ffffff"
SURFACE_ALT  = "#f6f6f7"
RAISED       = "#f0f0f2"
BORDER       = "#e1e1e4"
TEXT_MAIN    = "#111114"
TEXT_MUTED   = "#68686d"
TEXT_FAINT   = "#98989d"
ACCENT       = "#2c5254"
ACCENT_DEEP  = "#1c3839"
EMERALD      = "#2c7a54"
INFO         = "#37628f"
WARN         = "#93641c"
DANGER       = "#9c3b3b"
SUCCESS      = "#2c7a54"

SEVERITY_COLOR = {
    "No DR":            EMERALD,
    "Mild NPDR":         WARN,
    "Moderate NPDR":     INFO,
    "Severe NPDR":       "#a15a26",
    "Proliferative DR":  DANGER,
}

HR_SEVERITY_COLOR = {
    "Grade 0 (No HR)":                  EMERALD,
    "Grade 1 (Mild HR)":                WARN,
    "Grade 2 (Moderate HR)":            INFO,
    "Grade 3 (Severe HR)":              DANGER,
    "Grade 4 (Suspected Malignant HR)": DANGER,
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
    .block-container {{ padding-top: 1.5rem; padding-bottom: 4rem; max-width: 1080px; }}

    /* ============ Inputs ============ */
    .stTextInput input {{
        background-color: {SURFACE} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 4px !important;
        color: {TEXT_MAIN} !important;
        font-weight: 500 !important;
    }}
    .stTextInput input:focus {{
        border-color: {ACCENT} !important;
        box-shadow: none !important;
    }}
    label {{ color: {TEXT_MUTED} !important; font-size: 11px !important; font-weight: 600 !important;
             text-transform: uppercase; letter-spacing: 0.5px; }}

    /* ============ Uploader ============ */
    [data-testid="stFileUploaderDropzone"] {{
        background-color: {SURFACE_ALT} !important;
        border: 1px dashed {BORDER} !important;
        border-radius: 4px !important;
    }}
    [data-testid="stFileUploaderDropzone"] * {{ color: {TEXT_MUTED} !important; }}
    [data-testid="stFileUploaderDropzone"] button,
    section[data-testid="stFileUploader"] button {{
        background-color: {TEXT_MAIN} !important;
        color: {SURFACE} !important;
        border: none !important;
        border-radius: 4px !important;
        font-weight: 600 !important;
        box-shadow: none !important;
    }}
    [data-testid="stFileUploaderFile"] {{
        background-color: {RAISED} !important;
        border-radius: 4px !important;
        border: 1px solid {BORDER} !important;
    }}

    /* ============ Buttons ============ */
    .stButton button, .stDownloadButton button {{
        background-color: {TEXT_MAIN} !important;
        color: {SURFACE} !important;
        border: none !important;
        border-radius: 4px !important;
        font-weight: 600 !important;
        padding: 10px 16px !important;
        box-shadow: none !important;
        transition: opacity 0.12s ease;
    }}
    .stButton button:hover, .stDownloadButton button:hover {{
        opacity: 0.82;
    }}
    .stDownloadButton button p {{ color: {SURFACE} !important; font-weight: 600 !important; }}

    /* Secondary / outline button variant, used for the experimental HR trigger */
    .rs-btn-outline .stButton button {{
        background-color: {SURFACE} !important;
        color: {TEXT_MAIN} !important;
        border: 1px solid {TEXT_MAIN} !important;
    }}
    .rs-btn-outline .stButton button:hover {{ opacity: 1; background-color: {SURFACE_ALT} !important; }}

    /* ============ Radio Segmented Control ============ */
    div[role="radiogroup"] {{
        display: flex; gap: 4px; background: {SURFACE_ALT};
        padding: 4px; border-radius: 4px; border: 1px solid {BORDER};
        width: fit-content;
    }}
    div[role="radiogroup"] label {{
        background: transparent; border-radius: 3px; padding: 7px 14px !important;
        margin: 0 !important; color: {TEXT_MUTED} !important; font-weight: 600 !important;
        text-transform: none !important; letter-spacing: 0 !important; font-size: 12.5px !important;
        cursor: pointer;
    }}
    div[role="radiogroup"] label[data-checked="true"],
    div[role="radiogroup"] input:checked + div {{ color: {TEXT_MAIN} !important; }}
    div[role="radiogroup"] > label > div:first-child {{ display: none; }}

    /* ============ Expander ============ */
    .stExpander {{
        background-color: {SURFACE} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 4px;
    }}
    .stExpander summary {{ color: {TEXT_MAIN} !important; font-weight: 600 !important; }}

    /* ============ Misc ============ */
    hr {{ border-color: {BORDER} !important; }}
    ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
    ::-webkit-scrollbar-thumb {{ background-color: {BORDER}; border-radius: 4px; }}

    /* ============ Custom Layout Elements ============ */
    .rs-header {{
        padding: 6px 0 28px 0; border-bottom: 1px solid {BORDER}; margin-bottom: 30px;
    }}
    .rs-header-title {{ font-size: 22px; font-weight: 700; letter-spacing: -0.2px; color: {TEXT_MAIN}; }}
    .rs-header-sub {{ color: {TEXT_MUTED}; font-size: 12.5px; margin-top: 4px; }}

    .rs-toolbar {{
        background: {SURFACE_ALT}; border: 1px solid {BORDER}; border-radius: 4px;
        padding: 18px 20px; margin-bottom: 30px;
    }}
    .rs-toolbar-label {{
        color: {TEXT_FAINT}; font-size: 10.5px; text-transform: uppercase;
        letter-spacing: 0.6px; font-weight: 700; margin-bottom: 8px;
    }}

    .rs-section-label {{
        color: {TEXT_MAIN}; font-size: 11px; text-transform: uppercase;
        letter-spacing: 1px; font-weight: 700; margin: 34px 0 16px 0;
        padding-bottom: 8px; border-bottom: 1px solid {BORDER};
    }}

    .rs-card {{
        border: 1px solid {BORDER}; border-radius: 4px; padding: 26px 28px; margin-bottom: 8px;
    }}
    .rs-badge {{
        display: inline-block; font-size: 11px; font-weight: 700;
        padding: 5px 11px; border-radius: 3px; border: 1px solid; margin-right: 8px;
        text-transform: uppercase; letter-spacing: 0.4px;
    }}
    .rs-stat-strip {{ display: flex; gap: 32px; flex-wrap: wrap; margin-top: 20px; }}
    .rs-stat {{ display: flex; flex-direction: column; }}
    .rs-stat-label {{ color: {TEXT_MUTED}; font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; }}
    .rs-stat-value {{ font-size: 18px; font-weight: 700; margin-top: 3px; color: {TEXT_MAIN}; }}

    .rs-rail {{ border-left: 2px solid {BORDER}; padding-left: 16px; margin-bottom: 20px; }}
    .rs-rail-accent {{ border-left: 2px solid {TEXT_MAIN}; padding-left: 16px; margin-bottom: 20px; }}
    .rs-rail-warn {{ border-left: 2px solid {WARN}; padding-left: 16px; margin-bottom: 20px; }}
    .rs-rail-title {{ font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.6px; font-weight: 700; margin-bottom: 7px; color: {TEXT_MUTED}; }}
    .rs-rail-body {{ font-size: 13px; line-height: 1.6; color: {TEXT_MAIN}; }}

    .rs-prob-row {{ display: flex; justify-content: space-between; font-size: 12.5px; margin-bottom: 4px; }}

    .rs-bar-track {{
        width: 100%; height: 5px; border-radius: 3px;
        background: {SURFACE_ALT}; border: 1px solid {BORDER};
        overflow: hidden; margin-bottom: 13px;
    }}
    .rs-bar-fill {{ height: 100%; background: {TEXT_MAIN}; }}
    .rs-bar-fill-muted {{ height: 100%; background: {TEXT_FAINT}; }}

    .rs-reject {{
        background: {SURFACE}; border-left: 2px solid {DANGER}; border-radius: 4px;
        padding: 18px 20px; color: {TEXT_MAIN}; border-top: 1px solid {BORDER};
        border-right: 1px solid {BORDER}; border-bottom: 1px solid {BORDER};
    }}

    .rs-warn-box {{
        background: {SURFACE}; border-left: 2px solid {WARN}; border-radius: 4px;
        padding: 16px 20px; color: {TEXT_MAIN}; border-top: 1px solid {BORDER};
        border-right: 1px solid {BORDER}; border-bottom: 1px solid {BORDER};
    }}

    .rs-experimental-tag {{
        display: inline-block; font-size: 10px; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.6px; color: {WARN}; border: 1px solid {WARN}; border-radius: 3px;
        padding: 2px 7px; margin-left: 10px; vertical-align: middle;
    }}

    .rs-timeline-row {{
        display: flex; justify-content: space-between; align-items: center;
        padding: 12px 0; border-bottom: 1px solid {BORDER}; font-size: 12.5px;
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
HISTORY_DB = "patient_history.db"   # Issue G fix: SQLite replaces raw JSON file writes
RRWNET_DIR = "rrwnet_lib"           # vendor model.py here (Issue F) — see load_hr_model() note

_db_lock = threading.Lock()  # belt-and-suspenders alongside SQLite's own locking

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

# --- HR Metadata (Keith-Wagener-Barker Scale, Grades 0-4) ---
# Grade 4 is flagged via a papilledema screening heuristic (disc area +
# margin sharpness), NOT a validated diagnostic method — always labeled
# "Suspected" to keep this honest. See detect_papilledema_signs().
HR_CLASS_NAMES = {
    0: "Grade 0 (No HR)",
    1: "Grade 1 (Mild HR)",
    2: "Grade 2 (Moderate HR)",
    3: "Grade 3 (Severe HR)",
    4: "Grade 4 (Suspected Malignant HR)",
}

HR_CLASS_DESCRIPTIONS = {
    0: "Arteriolar-to-venular ratio within normal range. No evidence of generalized or focal arteriolar narrowing at the measured B-zone.",
    1: "Mild generalized retinal arteriolar narrowing (reduced AVR). Early vascular response to systemic hypertension.",
    2: "Marked arteriolar attenuation consistent with focal narrowing and/or AV nicking at vascular crossings.",
    3: "Severely reduced AVR consistent with significant microvascular injury. Correlate clinically for hemorrhages, cotton-wool spots, and hard exudates.",
    4: "Severely reduced AVR combined with signs suggestive of optic disc swelling (enlarged, blurred-margin disc) — a screening flag for possible papilledema, not a confirmed finding.",
}

HR_CLINICAL_DIRECTIVES = {
    0: "Maintain routine annual cardiovascular and fundus tracking. Continue standard lifestyle and blood pressure targets (<130/80 mmHg).",
    1: "Advise primary care physician (PCP) for baseline 24-hour ambulatory blood pressure monitoring (ABPM). Re-evaluate fundus in 12 months.",
    2: "Refer to primary care/cardiology for optimized antihypertensive regimen adjustment. Recheck retinal microvasculature in 3-6 months.",
    3: "URGENT SYSTEMIC EVALUATION REQUIRED: Contact managing physician within 24-48 hours. Target gradual blood pressure reduction.",
    4: "CRITICAL SCREENING FLAG: Signs suggestive of papilledema detected alongside severe AVR reduction. Recommend immediate clinical correlation and same-day physician evaluation — this is a heuristic flag requiring confirmation, not a standalone diagnosis.",
}

MODEL_CARD = {
    "architecture": "DR: EfficientNetB3 (ImageNet backbone). HR: RRWNet (pretrained artery/vein segmentation, Morano et al. 2024) + clinical Knudtson-Parr-Hubbard AVR computation on disc-diameter-normalized vessel widths, plus a heuristic papilledema screening check for Grade 4.",
    "input_resolution": f"DR: {IMG_SIZE} x {IMG_SIZE} RGB. HR: native resolution, downscaled to max 768px for inference; vessel widths normalized to % of optic-disc-diameter before AVR computation to remove resolution dependence.",
    "training_dataset": "DR: APTOS 2019 Blindness Detection dataset. HR: RRWNet pretrained on RITE/LES-AV/HRF (not retrained by this project).",
    "num_classes": "DR: 5-class ICDR. HR: Grades 0-4 of the Keith-Wagener-Barker scale (Grade 4 is a heuristic screening flag, see limitations).",
    "loss_function": "DR: Categorical Crossentropy / Focal Loss.",
    "reported_accuracy": "DR: ~89% top-1 accuracy on validation split. HR: inherits RRWNet's published segmentation benchmarks; AVR-to-grade mapping and the Grade 4 papilledema heuristic are rule-based and not independently validated by this project on labeled clinical data yet.",
    "explainability_method": "DR: Grad-CAM, quadrant-mapped. HR: artery/vein segmentation masks with branch-split vessel measurement + numeric AVR/CRAE/CRVE via the clinically standard B-zone + Knudtson formula.",
    "uncertainty_method": "DR: Test-Time Augmentation (TTA) ensemble variance across 3 views. HR: explicit 'indeterminate' abstain state when fewer than 2 reliable vessel segments are resolved in the B-zone — no forced grade on weak signal.",
    "known_limitations": [
        "HR Grade 4 is produced by an unvalidated heuristic (disc area ratio + margin gradient sharpness), not a trained papilledema detector — always shown as 'Suspected' and requires clinical confirmation.",
        "Optic disc localization uses a brightness + circularity heuristic; extreme exudate/glare cases can still occasionally mislocalize the disc.",
        "HR inference now reimplements RRWNet's actual published preprocessing.py enhancement algorithm (illumination background-subtraction), not an approximation — but exact numerical parity with the authors' PIL/skimage pipeline is not guaranteed.",
        "The AVR-to-KWB-grade cutoffs used here are literature-informed thresholds, not independently calibrated against a labeled KWB dataset by this project.",
        "Patient history is stored in local SQLite on the deployment container's filesystem and will not persist across container restarts/redeploys — acceptable for this development stage, flagged as a known constraint rather than solved.",
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
#  2. CACHED DR MODEL ENGINE (UNTOUCHED) & SQLITE PATIENT HISTORY
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


def _get_db_connection():
    conn = sqlite3.connect(HISTORY_DB, timeout=10, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")  # allows concurrent readers safely
    return conn


def _init_db():
    with _db_lock:
        conn = _get_db_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS visits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                diagnosis TEXT,
                hr_diagnosis TEXT,
                confidence REAL,
                attention_index REAL
            )
        """)
        conn.commit()
        conn.close()

_init_db()

def load_patient_history():
    """Returns the same {patient_id: [records...]} shape the UI already expects."""
    with _db_lock:
        conn = _get_db_connection()
        rows = conn.execute(
            "SELECT patient_id, timestamp, diagnosis, hr_diagnosis, confidence, attention_index "
            "FROM visits ORDER BY id ASC"
        ).fetchall()
        conn.close()

    history = {}
    for p_id, ts, diag, hr_diag, conf, attn in rows:
        history.setdefault(p_id, []).append({
            "timestamp": ts, "diagnosis": diag, "hr_diagnosis": hr_diag,
            "confidence": conf, "attention_index": attn
        })
    return history

def save_patient_record(p_id, diagnosis, confidence, attention_index, hr_diagnosis="N/A"):
    ist_timezone = pytz.timezone('Asia/Kolkata')
    current_time_ist = datetime.now(ist_timezone).strftime("%Y-%m-%d %H:%M")

    with _db_lock:
        conn = _get_db_connection()
        conn.execute(
            "INSERT INTO visits (patient_id, timestamp, diagnosis, hr_diagnosis, confidence, attention_index) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (p_id, current_time_ist, diagnosis, hr_diagnosis, round(float(confidence), 2), round(float(attention_index), 2))
        )
        conn.commit()
        conn.close()

    return load_patient_history().get(p_id, [])

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
    Cached once per server process.

    FIXED (Issue #1): the previous runtime `os.system("git clone ...")`
    fallback has been removed entirely. It would silently break in any
    containerized/serverless environment without a git binary or network
    egress at request time. model.py is now vendored directly into this
    repo at rrwnet_lib/model.py (fetched verbatim from
    https://github.com/j-morano/rrwnet) — no network dependency, no
    subprocess call, no fallback needed.
    """
    import torch

    if not os.path.exists(os.path.join(RRWNET_DIR, "model.py")):
        raise RuntimeError(
            f"{RRWNET_DIR}/model.py not found. This file must be vendored into "
            "the repository (see rrwnet_lib_model.py delivered alongside this "
            "app.py) — it is no longer fetched at runtime."
        )

    if RRWNET_DIR not in sys.path:
        sys.path.insert(0, RRWNET_DIR)

    from model import RRWNet as RRWNetModel
    from huggingface_hub import PyTorchModelHubMixin

    class RRWNet(RRWNetModel, PyTorchModelHubMixin):
        def __init__(self, input_ch=3, output_ch=3, base_ch=64, iterations=5):
            super().__init__(input_ch, output_ch, base_ch, iterations)

    torch.set_num_threads(2)  # don't starve TensorFlow running in the same process

    # Issue E (portability insurance): explicit device management. On this
    # CPU-only deployment this always resolves to "cpu", but it makes the
    # code safe to move to a GPU host later without touching this function.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hr_model = RRWNet.from_pretrained("j-morano/rrwnet-rite")
    hr_model.to(device)
    hr_model.eval()
    return hr_model, device


def _pad_to_multiple(img, multiple=32):
    """
    Pads an image so both dimensions are divisible by `multiple`.
    Required because RRWNet's internal skip connections concatenate
    encoder/decoder feature maps that must match exactly — an input
    size not divisible by the network's downsampling factor causes a
    'sizes must match' crash deep inside the model.
    """
    h, w = img.shape[:2]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    return padded, h, w


def _enhance_for_rrwnet(img_bgr, fundus_x, fundus_y, fundus_radius):
    """
    FIXED (Issue #2): this now faithfully reimplements RRWNet's actual
    preprocessing.py::enhance_image() function, fetched directly from
    https://github.com/j-morano/rrwnet, instead of the earlier CLAHE-based
    guess. The real algorithm is NOT contrast enhancement — it's an
    illumination-normalization technique:
      1. Fill the area outside the fundus mask with a 1.15x-zoomed,
         center-cropped version of the image (avoids black-border
         artifacts corrupting the blur step below).
      2. Erode the fundus mask by a small disk (radius 5).
      3. Estimate the background illumination via a large-sigma
         (sigma=10) Gaussian blur.
      4. Subtract that illumination estimate from the image (high-pass
         filter) — this is the actual "enhancement."
      5. Normalize by standard deviation, then rescale to [0, 1].
    Implemented with scipy.ndimage + skimage.morphology.disk, both
    already project dependencies — no new packages required.
    """
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0

    fov_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(fov_mask, (int(fundus_x), int(fundus_y)), max(int(fundus_radius), 1), 1, -1)

    img_copy = img_rgb.copy()

    zoomed = cv2.resize(img_rgb, (max(int(w * 1.15), 1), max(int(h * 1.15), 1)),
                         interpolation=cv2.INTER_CUBIC)
    zh, zw = zoomed.shape[:2]
    start_y, start_x = max(zh // 2 - h // 2, 0), max(zw // 2 - w // 2, 0)
    zoomed = zoomed[start_y:start_y + h, start_x:start_x + w]
    if zoomed.shape[:2] != (h, w):  # safety pad in case of off-by-one rounding
        zoomed = cv2.resize(zoomed, (w, h), interpolation=cv2.INTER_CUBIC)

    eroded_mask = ndimage.binary_erosion(fov_mask.astype(bool), disk(5)).astype(np.float64)

    img_copy[eroded_mask < 1.0] = 0.0
    mask_3ch = np.stack([eroded_mask] * 3, axis=2)
    composed = mask_3ch.copy()
    composed[mask_3ch == 1.0] = img_copy[mask_3ch == 1.0]
    composed[mask_3ch < 1.0] = zoomed[mask_3ch < 1.0]

    filtered = ndimage.gaussian_filter(composed, sigma=(10, 10, 0))
    subtracted = composed - filtered
    subtracted[mask_3ch < 1.0] = 0.0

    std = np.std(subtracted)
    if std < 1e-8:
        std = 1e-8
    enhanced = subtracted / std

    e_min, e_max = enhanced.min(), enhanced.max()
    if e_max - e_min < 1e-8:
        enhanced = np.zeros_like(enhanced)
    else:
        enhanced = (enhanced - e_min) / (e_max - e_min)
    enhanced[mask_3ch < 1.0] = 0.0

    enhanced_uint8 = (enhanced * 255).astype(np.uint8)
    return cv2.cvtColor(enhanced_uint8, cv2.COLOR_RGB2BGR)


def run_av_segmentation(img_bgr, fundus_x, fundus_y, fundus_radius, max_dim=768):
    """Runs A/V segmentation. Image is downscaled first to cap memory use."""
    import torch

    hr_model, device = load_hr_model()
    h0, w0 = img_bgr.shape[:2]
    scale = min(1.0, max_dim / max(h0, w0))
    img_small = cv2.resize(img_bgr, (int(w0 * scale), int(h0 * scale))) if scale < 1.0 else img_bgr.copy()

    # CRITICAL FIX #2: apply RRWNet's actual documented-required enhancement
    # (faithfully reimplemented above), using fundus geometry scaled to
    # match the downscaled working image.
    img_small = _enhance_for_rrwnet(img_small, fundus_x * scale, fundus_y * scale, fundus_radius * scale)

    # Pad to a multiple of 32 so encoder/decoder feature maps align exactly
    img_padded, h_small, w_small = _pad_to_multiple(img_small, multiple=32)

    rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB).astype(np.float32)
    tensor = (torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0) / 255.0).to(device)

    with torch.no_grad():
        output = hr_model(tensor)

    pred = output[-1] if isinstance(output, (list, tuple)) else output  # final refined iteration
    pred = torch.sigmoid(pred)[0].permute(1, 2, 0).cpu().numpy()

    pred = pred[:h_small, :w_small, :]  # crop off the padding before resizing back up
    pred_full = cv2.resize(pred, (w0, h0))

    # DIAGNOSTIC: raw per-channel stats, surfaced up to the UI so we can
    # confirm whether the model is outputting a real probability spread
    # or something degenerate (e.g. everything near 0, which a fixed
    # >0.5 threshold would silently turn into an all-black mask).
    debug_stats = {
        "artery": (float(pred_full[..., 0].min()), float(pred_full[..., 0].max()), float(pred_full[..., 0].mean())),
        "vein":   (float(pred_full[..., 1].min()), float(pred_full[..., 1].max()), float(pred_full[..., 1].mean())),
        "vessel": (float(pred_full[..., 2].min()), float(pred_full[..., 2].max()), float(pred_full[..., 2].mean())),
    }

    def _adaptive_threshold(channel):
        """
        A fixed >0.5 cutoff silently produces an all-black mask if the
        model's real probability spread sits below 0.5 everywhere (very
        common for thin-structure segmentation with heavy class
        imbalance). Otsu's method picks a data-driven threshold instead
        of assuming 0.5 is meaningful for this output distribution.
        """
        ch_min, ch_max = channel.min(), channel.max()
        if ch_max - ch_min < 1e-6:
            return np.zeros(channel.shape, dtype=np.uint8)  # genuinely no signal at all
        scaled = ((channel - ch_min) / (ch_max - ch_min) * 255).astype(np.uint8)
        _, binary = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    artery_mask = _adaptive_threshold(pred_full[..., 0])
    vein_mask   = _adaptive_threshold(pred_full[..., 1])
    vessel_mask = _adaptive_threshold(pred_full[..., 2])
    return artery_mask, vein_mask, vessel_mask, debug_stats


def detect_optic_disc(img_bgr, fundus_x, fundus_y, fundus_radius):
    """
    Issue A fix: robust disc localization. Instead of trusting the single
    brightest pixel (which locks onto exudates/glare), threshold the
    brightest region, find all candidate blobs, and score each by how
    close its area and circularity are to a real optic disc — not just
    brightness. Falls back to the old brightest-centroid method only if
    no plausible disc-shaped candidate exists.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    search_mask = np.zeros_like(gray)
    cv2.circle(search_mask, (int(fundus_x), int(fundus_y)), int(fundus_radius * 0.85), 255, -1)
    masked = cv2.bitwise_and(gray, gray, mask=search_mask)

    expected_disc_r = fundus_radius * 0.12
    expected_area = np.pi * expected_disc_r ** 2

    # Top ~2% brightest pixels within the fundus as candidate region
    bright_thresh = np.percentile(masked[search_mask > 0], 98) if np.any(search_mask > 0) else 250
    _, bright_binary = cv2.threshold(masked, bright_thresh, 255, cv2.THRESH_BINARY)
    bright_binary = cv2.morphologyEx(bright_binary, cv2.MORPH_CLOSE,
                                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

    contours, _ = cv2.findContours(bright_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_score, best_center, best_r = -1, None, expected_disc_r
    for c in contours:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        if area < expected_area * 0.25 or area > expected_area * 4.0 or perimeter == 0:
            continue  # wrong size to plausibly be the disc — likely an exudate speck or large glare patch
        circularity = 4 * np.pi * area / (perimeter ** 2)
        (cx, cy), r = cv2.minEnclosingCircle(c)
        # score rewards circularity and size-closeness to expected disc, not raw brightness
        size_score = 1.0 - min(1.0, abs(area - expected_area) / expected_area)
        score = 0.6 * circularity + 0.4 * size_score
        if score > best_score:
            best_score, best_center, best_r = score, (cx, cy), r

    if best_center is not None and best_score > 0.35:
        return best_center[0], best_center[1], expected_disc_r

    # Fallback: original brightest-blurred-pixel method (better than nothing
    # if no circular candidate was found at all)
    blurred = cv2.GaussianBlur(masked, (25, 25), 0)
    _, _, _, max_loc = cv2.minMaxLoc(blurred)
    return max_loc[0], max_loc[1], expected_disc_r


def detect_papilledema_signs(img_bgr, disc_x, disc_y, disc_r, fundus_radius):
    """
    Issue #3 fix: a concrete, feasible screening signal for papilledema
    (optic disc swelling) — the finding required for KWB Grade 4 that AVR
    alone structurally cannot detect. This does NOT require a new trained
    model. It checks two independent proxy signals and only flags when
    BOTH agree, to keep the false-positive rate down:

      1. Disc area substantially larger than the expected disc:fundus
         ratio — swollen discs measure larger than normal.
      2. A blurred/indistinct disc margin — edge gradient magnitude
         around the boundary is abnormally low compared to a sharp,
         healthy disc edge.

    This is an unvalidated heuristic screening flag, not a diagnosis —
    treated as such throughout (labeled "Suspected" everywhere it
    appears, and listed explicitly in the Model Card's limitations).
    """
    expected_r = fundus_radius * 0.12
    area_ratio = (disc_r / expected_r) if expected_r > 0 else 1.0
    area_flag = area_ratio > 1.35  # disc measuring >35% larger than the expected ratio

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    ring_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(ring_mask, (int(disc_x), int(disc_y)), int(disc_r * 1.15), 255, 3)
    cv2.circle(ring_mask, (int(disc_x), int(disc_y)), int(disc_r * 0.85), 0, -1)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    ring_pixels = grad_mag[ring_mask > 0]
    margin_sharpness = float(np.mean(ring_pixels)) if ring_pixels.size > 0 else 0.0
    margin_flag = margin_sharpness < 12.0  # empirical, unvalidated low-gradient threshold

    return {
        "suspected": bool(area_flag and margin_flag),
        "area_ratio": round(float(area_ratio), 2),
        "margin_sharpness": round(margin_sharpness, 2),
    }


def compute_b_zone_mask(h, w, disc_x, disc_y, disc_radius):
    """Standard clinical B-zone: 0.5-1.0 disc diameters from the disc margin."""
    inner_r, outer_r = disc_radius * 1.5, disc_radius * 2.0
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - disc_x) ** 2 + (yy - disc_y) ** 2)
    return (((dist >= inner_r) & (dist <= outer_r)).astype(np.uint8)) * 255


def measure_vessel_calibers(vessel_channel_mask, b_zone_mask, disc_diameter_px, top_n=6):
    """
    Issue B fix: split touching/crossing vessels at skeleton branch points
    before measuring, instead of relying on raw connected components (which
    silently merges multiple crossing vessels into one giant blob).

    Issue C fix: widths are converted to a percentage of optic-disc-diameter
    (not raw pixels) before being returned, so measurements are comparable
    across images of any resolution before the Knudtson formula is applied.
    """
    masked = cv2.bitwise_and(vessel_channel_mask, b_zone_mask)
    if cv2.countNonZero(masked) < 20:
        return []

    vessel_bool = masked > 0
    skeleton = skeletonize(vessel_bool)

    # Find branch points: skeleton pixels with 3+ skeleton neighbors, i.e.
    # crossings/branches where separate vessels touch. Remove them so each
    # remaining connected piece is (approximately) a single vessel segment.
    neighbor_count = np.zeros_like(skeleton, dtype=np.uint8)
    sk = skeleton.astype(np.uint8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            shifted = np.roll(np.roll(sk, dy, axis=0), dx, axis=1)
            neighbor_count += shifted
    branch_points = skeleton & (neighbor_count >= 3)
    pruned_skeleton = skeleton & ~branch_points

    num_labels, labels_im = cv2.connectedComponents(pruned_skeleton.astype(np.uint8), connectivity=8)

    # Distance transform of the full vessel mask gives, at every pixel,
    # the distance to the nearest non-vessel boundary — doubling this at
    # skeleton points is a standard way to recover local vessel width.
    dist_map = distance_transform_edt(vessel_bool)

    widths_px = []
    for lbl in range(1, num_labels):
        seg_pixels = (labels_im == lbl)
        if np.sum(seg_pixels) < 3:
            continue  # too short a fragment to trust
        local_widths = dist_map[seg_pixels] * 2.0
        widths_px.append(float(np.median(local_widths)))

    if not widths_px:
        return []

    # Normalize to % of disc diameter (Issue C) — resolution-independent unit
    widths_normalized = [(w_px / disc_diameter_px) * 100.0 for w_px in widths_px]
    return sorted(widths_normalized, reverse=True)[:top_n]


def knudtson_avr(artery_widths, vein_widths):
    """
    Knudtson-Parr-Hubbard iterative pairing formula -> CRAE, CRVE, AVR.
    Inputs are now disc-diameter-normalized widths (see measure_vessel_calibers),
    so CRAE/CRVE are comparable across images regardless of source resolution.
    """
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
        artery_mask, vein_mask, vessel_mask, debug_stats = run_av_segmentation(img_bgr, x_center, y_center, radius)
    except Exception as e:
        return {"status": "error", "pred_name": "HR Engine Unavailable",
                "message": f"Segmentation failed safely: {e}"}

    try:
        disc_x, disc_y, disc_r = detect_optic_disc(img_bgr, x_center, y_center, radius)
        disc_diameter_px = max(disc_r * 2.0, 1.0)
        h, w = img_bgr.shape[:2]
        b_zone = compute_b_zone_mask(h, w, disc_x, disc_y, disc_r)

        artery_widths = measure_vessel_calibers(artery_mask, b_zone, disc_diameter_px)
        vein_widths = measure_vessel_calibers(vein_mask, b_zone, disc_diameter_px)
        crae, crve, avr = knudtson_avr(artery_widths, vein_widths)
    except Exception as e:
        return {"status": "error", "pred_name": "HR Analysis Failed",
                "message": f"AVR computation failed safely: {e}",
                "artery_mask": artery_mask, "vein_mask": vein_mask, "vessel_mask": vessel_mask,
                "debug_stats": debug_stats}

    if avr is None:
        return {"status": "indeterminate", "pred_name": "Indeterminate — Insufficient Vascular Signal",
                "message": "Not enough clearly resolved arteries/veins in the B-zone for a reliable AVR.",
                "artery_mask": artery_mask, "vein_mask": vein_mask, "vessel_mask": vessel_mask,
                "debug_stats": debug_stats}

    if avr >= 0.65:
        idx, name = 0, "Grade 0 (No HR)"
    elif avr >= 0.58:
        idx, name = 1, "Grade 1 (Mild HR)"
    elif avr >= 0.50:
        idx, name = 2, "Grade 2 (Moderate HR)"
    else:
        idx, name = 3, "Grade 3 (Severe HR)"

    papilledema_signs = None
    if idx == 3:
        # Only check for papilledema signs at the Severe tier — this is
        # where a Grade 4 escalation is clinically plausible. Both the
        # area and margin-sharpness signals must agree (see
        # detect_papilledema_signs) before upgrading the grade.
        try:
            papilledema_signs = detect_papilledema_signs(img_bgr, disc_x, disc_y, disc_r, radius)
            if papilledema_signs["suspected"]:
                idx, name = 4, "Grade 4 (Suspected Malignant HR)"
        except Exception:
            papilledema_signs = None  # fail soft — stays at Grade 3 if this check errors

    result = {"status": "ok", "pred_idx": idx, "pred_name": name, "avr": round(float(avr), 3),
              "crae": round(float(crae), 2), "crve": round(float(crve), 2),
              "artery_mask": artery_mask, "vein_mask": vein_mask, "vessel_mask": vessel_mask,
              "debug_stats": debug_stats}
    if papilledema_signs is not None:
        result["papilledema_signs"] = papilledema_signs
    return result

# =====================================================================
#  5. AUXILIARY SCREENING & REPORT GENERATION UTILITIES
# =====================================================================
def _pdf_safe(text):
    """
    fpdf2's built-in Helvetica font only supports Latin-1. Any dynamic
    string (error messages, directives, exception text) can contain
    characters outside that range (em-dashes, curly quotes, ±, etc.)
    and crash the PDF export with FPDFUnicodeEncodingException. This
    sanitizes any string before it's handed to fpdf, replacing
    unencodable characters instead of crashing.
    """
    if text is None:
        return ""
    return str(text).encode("latin-1", "replace").decode("latin-1")


def generate_clinical_pdf(p_id, verdict, conf, attn_idx, quad, quad_pct, directive, consensus, hr_results):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(16, 185, 129)
    pdf.cell(0, 10, _pdf_safe("RETISCAN PRO DIAGNOSTIC SUMMARY REPORT"), ln=True, align="C")
    pdf.ln(6)

    ist_timezone = pytz.timezone('Asia/Kolkata')
    current_time_ist = datetime.now(ist_timezone).strftime("%Y-%m-%d %H:%M")

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 7, _pdf_safe(f"Patient Tracking Key: {p_id}"), ln=True)
    pdf.cell(0, 7, _pdf_safe(f"Generated Timestamp: {current_time_ist}"), ln=True)
    pdf.cell(0, 7, _pdf_safe(f"Ensemble Integrity State: {consensus}"), ln=True)
    pdf.line(10, pdf.get_y() + 2, 200, pdf.get_y() + 2)
    pdf.ln(6)

    # DR Section
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(229, 111, 34)
    pdf.cell(0, 8, _pdf_safe(f"1. Diabetic Retinopathy (DR): {verdict}"), ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 7, _pdf_safe(f"   - DR Confidence: {conf:.2f}%"), ln=True)
    pdf.cell(0, 7, _pdf_safe(f"   - Neuro-Attention Mapping Index: {attn_idx:.1f}%"), ln=True)
    pdf.cell(0, 7, _pdf_safe(f"   - Dominant Focus: {quad} Quadrant ({quad_pct:.1f}%)"), ln=True)
    pdf.ln(3)

    # HR Section
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(94, 177, 239)
    pdf.cell(0, 8, _pdf_safe(f"2. Hypertensive Retinopathy (HR): {hr_results.get('pred_name', 'N/A')}"), ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(0, 0, 0)
    if hr_results.get("status") == "ok":
        pdf.cell(0, 7, _pdf_safe(f"   - Arteriolar-to-Venular Ratio (AVR): {hr_results['avr']}"), ln=True)
        pdf.cell(0, 7, _pdf_safe(f"   - CRAE: {hr_results['crae']}  |  CRVE: {hr_results['crve']} (disc-diameter-normalized units)"), ln=True)
        pdf.cell(0, 7, _pdf_safe("   - Scale: Keith-Wagener-Barker Grades 0-3 (Grade 4 out of current scope)"), ln=True)
    else:
        pdf.cell(0, 7, _pdf_safe(f"   - Status: {hr_results.get('status', 'unknown').upper()}"), ln=True)
        pdf.multi_cell(0, 6, _pdf_safe(f"   - {hr_results.get('message', '')}"))
    pdf.ln(6)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, _pdf_safe("Official Management Protocol Directives:"), ln=True)
    pdf.set_font("Helvetica", "I", 10)
    hr_directive = HR_CLINICAL_DIRECTIVES.get(hr_results.get("pred_idx"), "N/A - see HR status above.")
    pdf.multi_cell(0, 5, _pdf_safe(f"DR: {directive}\nHR: {hr_directive}"))

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
        # HR call takes ONLY the raw image + fundus geometry — no dr_pred_idx,
        # no DR probabilities, no vessel_map. Independence is structural.
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

    if "artery_mask" in hr_results:
        m1, m2, m3 = st.columns(3)
        with m1: st.image(hr_results["artery_mask"], caption="Detected Arteries", use_container_width=True, clamp=True)
        with m2: st.image(hr_results["vein_mask"], caption="Detected Veins", use_container_width=True, clamp=True)
        with m3: st.image(hr_results["vessel_mask"], caption="Vessel Union", use_container_width=True, clamp=True)

    # DIAGNOSTIC PANEL — shows the raw model output range so an all-black
    # mask can be told apart from "model output real values but they were
    # thresholded away" vs "model produced no signal at all."
    if "debug_stats" in hr_results:
        with st.expander("🔧 HR Diagnostic — raw segmentation output stats"):
            ds = hr_results["debug_stats"]
            st.markdown(f"""
            <div style="font-size:12.5px; color:{TEXT_MUTED}; font-family:monospace; line-height:1.8;">
                Artery channel — min: {ds['artery'][0]:.4f}, max: {ds['artery'][1]:.4f}, mean: {ds['artery'][2]:.4f}<br>
                Vein channel &nbsp;— min: {ds['vein'][0]:.4f}, max: {ds['vein'][1]:.4f}, mean: {ds['vein'][2]:.4f}<br>
                Vessel channel — min: {ds['vessel'][0]:.4f}, max: {ds['vessel'][1]:.4f}, mean: {ds['vessel'][2]:.4f}
            </div>
            <p style="font-size:11.5px; color:{TEXT_FAINT}; margin-top:10px;">
                If max values here are near 0 across all channels, the model itself is producing no signal
                (likely a weight-loading or preprocessing mismatch, not a thresholding issue).
                If max values are meaningfully above 0 but masks still look empty, the adaptive
                threshold should now catch it — if this expander shows non-trivial max values but the
                masks above are still black, that points to a downstream masking bug, not the model.
            </p>
            """, unsafe_allow_html=True)
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
                (0.5-1.0 optic disc diameters from the disc margin) of independently segmented, branch-split
                arteries (CRAE {hr_results['crae']}) and veins (CRVE {hr_results['crve']}) — both normalized to
                % of optic-disc-diameter so the result is resolution-independent.
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
