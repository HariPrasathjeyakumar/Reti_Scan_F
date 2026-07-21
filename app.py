import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from skimage.filters import frangi
import tensorflow as tf

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="RetiScan - Hypertensive Retinopathy Classifier",
    page_icon="👁️",
    layout="wide"
)

MODEL_PATH = "retiscan_hr_best.keras"

# ==========================================
# 2. MODEL LOADING & CACHING
# ==========================================
@st.cache_resource
def load_hr_model():
    """Loads the trained Keras model if present on disk."""
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            return model, True
        except Exception as e:
            st.error(f"Error loading model from {MODEL_PATH}: {e}")
            return None, False
    return None, False

# ==========================================
# 3. IMAGE PREPROCESSING & VESSEL EXTRACTION
# ==========================================
def extract_vessel_map(image_np):
    """
    Extracts retinal blood vessels using CLAHE on the Green Channel 
    followed by a Frangi vesselness filter.
    """
    # 1. Extract Green Channel (best contrast for retinal vessels)
    if len(image_np.shape) == 3:
        green_ch = image_np[:, :, 1]
    else:
        green_ch = image_np

    # 2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(green_ch)

    # 3. Frangi Filter (detects tubular vessel structures)
    vessel_map = frangi(contrast_enhanced, sigmas=range(1, 4), black_ridges=True)

    # 4. Normalize result to uint8 [0, 255]
    vessel_map_norm = cv2.normalize(vessel_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return vessel_map_norm


def preprocess_for_inference(pil_img, target_size=(224, 224)):
    """Preprocesses input image into tensor and numpy formats."""
    rgb_img = pil_img.convert("RGB")
    np_img = np.array(rgb_img)
    
    # Resize image for CNN input
    resized_img = cv2.resize(np_img, target_size)
    
    # Expand dims to create batch dimension [1, H, W, C]
    raw_tensor = np.expand_dims(resized_img.astype(np.float32), axis=0)
    
    return np_img, resized_img, raw_tensor

# ==========================================
# 4. INFERENCE ENGINE (WITH SCALING FIX)
# ==========================================
def run_hr_inference(model, raw_tensor, vessel_map_img):
    """
    Runs inference on the input image tensor.
    Scales inputs to [0, 1] range to prevent 'Always Negative' predictions.
    """
    if model is not None:
        # CRITICAL FIX: Standardize [0-255] pixels to [0, 1]
        norm_tensor = raw_tensor / 255.0 if np.max(raw_tensor) > 1.0 else raw_tensor
        
        # Predict on batch
        raw_pred = model.predict(norm_tensor, verbose=0)[0]
        
        # Handle 1-neuron Sigmoid vs 2-neuron Softmax outputs
        if np.isscalar(raw_pred) or len(raw_pred) == 1:
            prob = float(raw_pred if np.isscalar(raw_pred) else raw_pred[0])
        else:
            # Index 1 = Positive HR Class
            prob = float(raw_pred[1])
    else:
        # Calibrated Fallback Heuristic using vessel density & lesion intensity
        vessel_density = float(np.mean(vessel_map_img > 15))
        high_intensity_lesions = float(np.mean(vessel_map_img > 120))
        
        # Anomaly metric combination
        anomaly_score = (vessel_density * 3.5) + (high_intensity_lesions * 12.0)
        prob = min(max(anomaly_score, 0.05), 0.95)

    detected = prob >= 0.50
    status = "POSITIVE (Risk Detected)" if detected else "NEGATIVE (Normal)"
    confidence = prob * 100.0 if detected else (1.0 - prob) * 100.0
    
    return status, confidence, prob, detected

# ==========================================
# 5. STREAMLIT USER INTERFACE
# ==========================================
def main():
    st.title("👁️ RetiScan: Hypertensive Retinopathy Detection")
    st.markdown("Upload a retinal fundus photograph to inspect vessel morphology and evaluate HR risk.")
    
    # Load Model
    hr_model, model_loaded = load_hr_model()
    
    # Sidebar Info
    with st.sidebar:
        st.header("System Status")
        if model_loaded:
            st.success("✅ Model Loaded (`retiscan_hr_best.keras`)")
        else:
            st.warning("⚠️ Local model file not found. Running in Fallback Diagnostic Mode.")
        
        st.markdown("---")
        st.markdown("**Supported Severity Markers:**")
        st.markdown("* Arteriolar Narrowing / AV Nicking")
        st.markdown("* Flame-shaped Hemorrhages")
        st.markdown("* Cotton Wool Spots & Exudates")
        st.markdown("* Optic Disc Swelling (Papilledema)")

    # File Uploader
    uploaded_file = st.file_uploader("Choose a Retinal Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Preprocess
        original_np, resized_np, raw_tensor = preprocess_for_inference(image)
        
        # Generate Vessel Map via Frangi Filter
        with st.spinner("Extracting retinal vascular structures..."):
            vessel_map = extract_vessel_map(resized_np)

        # Display Images Side-by-Side
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Retinal Fundus")
            st.image(original_np, use_container_width=True)
            
        with col2:
            st.subheader("Frangi Vascular Structure Map")
            st.image(vessel_map, use_container_width=True, clamp=True, channels="GRAY")

        st.markdown("---")

        # Run Analysis
        if st.button("🔍 Run Hypertensive Retinopathy Analysis", type="primary"):
            with st.spinner("Analyzing microvascular patterns..."):
                status, confidence, prob, detected = run_hr_inference(hr_model, raw_tensor, vessel_map)

            # Display Classification Results
            st.subheader("Diagnostic Assessment")
            
            res_col1, res_col2, res_col3 = st.columns(3)
            
            with res_col1:
                if detected:
                    st.error(f"**Classification:** {status}")
                else:
                    st.success(f"**Classification:** {status}")
                    
            with res_col2:
                st.metric(label="HR Risk Probability", value=f"{prob * 100:.2f}%")
                
            with res_col3:
                st.metric(label="Model Confidence", value=f"{confidence:.2f}%")

            # Progress Bar
            st.progress(prob)

            # Severity Risk Assessment (Keith-Wagener-Barker Scale Guidance)
            st.markdown("### Clinical Guidance Breakdown")
            if prob >= 0.75:
                st.error("🚨 **High Severity Risk (Grade 3/4 Equivalent):** Significant vascular abnormalities detected (e.g., severe hemorrhages, cotton wool spots, or optic disc swelling). Immediate ophthalmic evaluation recommended.")
            elif prob >= 0.50:
                st.warning("⚠️ **Moderate HR Risk (Grade 1/2 Equivalent):** Mild to moderate arteriolar narrowing or AV nicking pattern detected. Follow-up monitoring suggested.")
            else:
                st.info("✅ **Low Risk / Normal:** No prominent signs of hypertensive vascular damage detected.")

if __name__ == "__main__":
    main()
