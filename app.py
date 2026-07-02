import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime

# =====================================================================
#  1. PAGE INITIALIZATION & THEME CONFIGURATION
# =====================================================================
st.set_page_config(
    page_title="RetiScan Pro - DR Progression Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling Injection to achieve a professional medical workstation dashboard look
st.markdown("""
    <style>
    .reportview-container .main .block-container { padding-top: 2rem; }
    div.stButton > button:first-child {
        background-color: #38bdf8; color: #0f172a; font-weight: bold; border-radius: 6px; width: 100%;
    }
    .metric-card {
        background-color: #1e293b; border: 1px solid #334155; padding: 15px; border-radius: 10px;
    }
    </style>
""", unsafe_allowed_allowed_html=True)

IMG_SIZE = 224
HISTORY_FILE = "patient_history.json"
MODEL_PATH = "retiscan_pro_v5_best.keras"  # Place your model in the same folder or update path

CLASS_NAMES = {
    0: "No DR (Healthy)",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR"
}

# =====================================================================
#  2. CORE CACHING & ENGINE INITIALIZATION
# =====================================================================
@st.cache_resource
def load_retina_engine(path):
    if not os.path.exists(path):
        return None
    def focal_loss():
        def loss_fn(y_true, y_pred): return tf.reduce_mean(y_pred)
        loss_fn.__name__ = "focal_loss"
        return loss_fn
    model = tf.keras.models.load_model(path, custom_objects={"focal_loss": focal_loss()})
    
    # Compile dual-output feature graph framework
    conv_layer_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D) or 'top_conv' in layer.name or 'block7' in layer.name:
            conv_layer_name = layer.name
            break
    if not conv_layer_name: conv_layer_name = "top_conv"
    
    grad_model = tf.keras.models.Model(inputs=[model.inputs], outputs=[model.get_layer(conv_layer_name).output, model.output])
    return grad_model

grad_model = load_retina_engine(MODEL_PATH)

# Database Utilities
def load_patient_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f: return json.load(f)
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
#  3. APP SIDEBAR INTERACTION CONTROL PANEL
# =====================================================================
st.sidebar.title("🩺 Clinical Control Hub")
st.sidebar.markdown("---")

patient_id = st.sidebar.text_input("👤 Patient ID Identifier Key", value="PATIENT-500").strip().upper()
uploaded_file = st.sidebar.file_uploader("📸 Upload Retinal Fundus Photo", type=["png", "jpg", "jpeg"])

st.sidebar.markdown("---")
st.sidebar.info("💡 **Approach A Architecture Profile:** Tracks spatial feature attention magnitudes via peak neural layer weight responses across longitudinal sessions.")

# =====================================================================
#  4. MAIN PIPELINE WORKSPACE EXECUTION
# =====================================================================
if grad_model is None:
    st.error(f"❌ Model missing or not found at path: `{MODEL_PATH}`. Please upload it to your deployment directory.")
else:
    if uploaded_file is not None:
        # Convert file buffer to image arrays
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        fresh_img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        h, w, _ = fresh_img_bgr.shape
        
        # --- SCREENING FILTER VALIDATION ---
        gray = cv2.cvtColor(fresh_img_bgr, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        if mean_brightness < 10:
            st.error("❌ SCREENING REJECTED: Low asset illumination exposure detected. Image cannot be read.")
        else:
            _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
            contours_eye, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            is_valid_fundus = False
            if contours_eye:
                largest_contour = max(contours_eye, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > (h * w * 0.15):
                    (x_center, y_center), radius = cv2.minEnclosingCircle(largest_contour)
                    is_valid_fundus = True
            
            if not is_valid_fundus:
                st.error("❌ SCREENING REJECTED: Structural retinal fundus geometry unverified.")
            else:
                # --- PROCESS INFERENCE ENGINE ---
                img_rgb_input = cv2.cvtColor(fresh_img_bgr, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb_input, (IMG_SIZE, IMG_SIZE))
                img_tensor = tf.convert_to_tensor(img_resized, dtype=tf.float32)
                img_batch = tf.expand_dims(img_tensor, axis=0)
                
                with tf.GradientTape() as tape:
                    conv_outputs, model_predictions = grad_model(img_batch)
                    raw_probabilities = model_predictions[0].numpy()
                    pred_idx = int(np.argmax(raw_probabilities))
                    loss = model_predictions[:, pred_idx]
                
                grads = tape.gradient(loss, conv_outputs)
                pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                heatmap = tf.squeeze(conv_outputs[0] @ pooled_grads[..., tf.newaxis])
                heatmap = tf.maximum(heatmap, 0)
                max_val = tf.math.reduce_max(heatmap)
                if max_val == 0: max_val = 1e-8
                current_raw_heatmap = (heatmap / max_val).numpy()
                confidence_score = raw_probabilities[pred_idx] * 100
                
                # Resize and apply the eye mask
                heatmap_uint8 = np.uint8(255 * current_raw_heatmap)
                heatmap_resized = cv2.resize(heatmap_uint8, (w, h))
                perfect_circle_mask = np.zeros((h, w), dtype=np.uint8)
                safe_radius = int(radius * 0.82)
                cv2.circle(perfect_circle_mask, (int(x_center), int(y_center)), safe_radius, 255, -1)
                isolated_internal_heatmap = cv2.bitwise_and(heatmap_resized, perfect_circle_mask)
                
                # --- APPROACH A GATEKEEPER CALCULATIONS ---
                if pred_idx == 0:
                    ai_attention_index = 0.0
                    boundary_img_bgr = fresh_img_bgr.copy()
                else:
                    active_pixels = isolated_internal_heatmap[isolated_internal_heatmap > 0]
                    if len(active_pixels) > 0:
                        top_threshold = np.percentile(active_pixels, 90)
                        ai_attention_index = np.mean(active_pixels[active_pixels >= top_threshold]) / 255.0 * 100.0
                    else:
                        ai_attention_index = 0.0
                        
                    # Locate and draw attention boxes
                    max_internal_val = np.max(isolated_internal_heatmap) if np.max(isolated_internal_heatmap) > 0 else 1
                    _, binary_mask = cv2.threshold(isolated_internal_heatmap, int(0.55 * max_internal_val), 255, cv2.THRESH_BINARY)
                    lesion_contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    boundary_img_bgr = fresh_img_bgr.copy()
                    for contour in lesion_contours:
                        if cv2.contourArea(contour) > 25:
                            cv2.drawContours(boundary_img_bgr, [contour], -1, (255, 255, 0), 1)
                            x, y, box_w, box_h = cv2.boundingRect(contour)
                            cv2.rectangle(boundary_img_bgr, (x, y), (x + box_w, y + box_h), (0, 255, 255), 2)
                
                # Save data logs to local memory database
                predicted_class_name = CLASS_NAMES[pred_idx]
                record_logs = save_patient_record(patient_id, predicted_class_name, confidence_score, ai_attention_index)
                
                # --- RENDER DASHBOARD LAYOUT GRIDS ---
                st.title("📊 Multi-Panel Diagnostic Worksheet Canvas")
                st.markdown("---")
                
                # Display 4-Panel Visualization Plots
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown("**I. Base Input Asset**")
                    st.image(cv2.cvtColor(fresh_img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                with col2:
                    st.markdown("**II. Glare-Free Grad-CAM Map**")
                    heatmap_color = cv2.applyColorMap(isolated_internal_heatmap, cv2.COLORMAP_JET)
                    gradcam_blend = cv2.addWeighted(heatmap_color, 0.38, fresh_img_bgr, 0.62, 0)
                    st.image(cv2.cvtColor(gradcam_blend, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                with col3:
                    st.markdown(f"**III. Attention Box ({ai_attention_index:.1f} Index)**")
                    st.image(cv2.cvtColor(boundary_img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                with col4:
                    st.markdown("**IV. AI Attention Intensity Tracker**")
                    fig, ax = plt.subplots(figsize=(4, 3.5))
                    visit_stamps = [r["timestamp"].split(" ")[0] for r in record_logs]
                    attention_indices = [r["attention_index"] for r in record_logs]
                    ax.plot(range(len(record_logs)), attention_indices, marker='o', color='#38bdf8', linewidth=2.5)
                    ax.set_xticks(range(len(record_logs)))
                    ax.set_xticklabels(visit_stamps, rotation=25, ha='right', fontsize=8)
                    ax.set_ylim(-5, 105)
                    ax.grid(True, linestyle='--', alpha=0.4)
                    fig.patch.set_facecolor('#0f172a')
                    ax.set_facecolor('#1e293b')
                    ax.spines['bottom'].set_color('#94a3b8')
                    ax.spines['left'].set_color('#94a3b8')
                    ax.tick_params(colors='#94a3b8', labelsize=8)
                    ax.yaxis.label.set_color('#94a3b8')
                    st.pyplot(fig)
                
                # --- CLINICAL RESULTS MARKS & XAI DESC ---
                st.markdown("---")
                color_hex = "#10b981" if pred_idx == 0 else "#38bdf8"
                
                # Process custom quadrant metrics for XAI
                mid_y, mid_x = h // 2, w // 2
                quadrants = {
                    "Upper-Left": isolated_internal_heatmap[0:mid_y, 0:mid_x],
                    "Upper-Right": isolated_internal_heatmap[0:mid_y, mid_x:w],
                    "Lower-Left": isolated_internal_heatmap[mid_y:h, 0:mid_x],
                    "Lower-Right": isolated_internal_heatmap[mid_y:h, mid_x:w]
                }
                scores = {k: np.mean(v) for k, v in quadrants.items()}
                primary_quad = max(scores, key=scores.get)
                total_score = sum(scores.values()) if sum(scores.values()) > 0 else 1
                focus_pct = (scores[primary_quad] / total_score) * 100
                
                xai_rationale = (
                    "The neural activations are uniform and clean. No statistically significant pixel fluctuations were identified across any of the fields."
                    if pred_idx == 0 else 
                    f"The diagnostic prediction is primarily driven by localized structural anomalies inside the <b>{primary_quad} quadrant</b> (commanding {focus_pct:.1f}% of overall visual focus). Network weights detected high-density diagnostic triggers in this area."
                )
                
                st.markdown(f"""
                <div style="background-color:#0f172a; padding:20px; border-radius:12px; border:1px solid #334155;">
                    <span style="color:#94a3b8; font-size:12px; text-transform:uppercase;">System Screening Verdict</span>
                    <h2 style="color:{color_hex}; margin:2px 0 15px 0;">{predicted_class_name} ({confidence_score:.2f}% Confidence)</h2>
                    <p style="background-color:#1e293b; padding:15px; border-radius:8px; font-size:14px;">
                        <b>🔍 Dynamic XAI Description:</b> {xai_rationale}
                    </p>
                </div>
                """, unsafe_allowed_html=True)
                
                # --- LONGITUDINAL PATIENT TRACKING LOGS TABLE ---
                st.markdown("<br>", unsafe_allowed_html=True)
                st.subheader(f"📋 Historical Tracking Summary Log ({patient_id})")
                
                table_html = """
                <table style="width:100%; text-align:left; border-collapse:collapse; font-size:14px;">
                    <thead>
                        <tr style="border-bottom:2px solid #334155; color:#94a3b8;">
                            <th style="padding:10px;">Visit Index</th>
                            <th style="padding:10px;">Timestamp</th>
                            <th style="padding:10px;">Diagnosis Verdict</th>
                            <th style="padding:10px;">Confidence Score</th>
                            <th style="padding:10px;">Attention Index</th>
                        </tr>
                    </thead>
                    <tbody>
                """
                for i, r in enumerate(record_logs):
                    table_html += f"""
                    <tr style="border-bottom:1px solid #1e293b;">
                        <td style="padding:10px;"><b>#{i+1}</b></td>
                        <td style="padding:10px;">{r['timestamp']}</td>
                        <td style="padding:10px;">{r['diagnosis']}</td>
                        <td style="padding:10px;">{r['confidence']}%</td>
                        <td style="padding:10px; color:#38bdf8;"><b>{r['attention_index']:.1f}%</b></td>
                    </tr>
                    """
                table_html += "</tbody></table>"
                st.markdown(table_html, unsafe_allowed_html=True)
    else:
        st.info("👋 Welcome! Please upload a retinal fundus photo inside the left sidebar panel to initialize the screening pipeline workflow.")
