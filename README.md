RetiScan Pro v5: Dual-Screening Retinal AI System

📌 Overview

RetiScan Pro is designed for primary care clinics, rural health centers, and mobile screening units. It combines two specialized deep learning models to perform dual-disease analysis on a single retinal photo, automatically filtering bad-quality images, measuring microvascular changes, and generating instant visual PDF diagnostic reports.

✨ Key Features

Dual Disease Screening:

Diabetic Retinopathy (DR): 5-stage ICDR classification using EfficientNetB3.

Hypertensive Retinopathy (HR): Automated vessel width measurement (Arteriolar-to-Venular Ratio / AVR) using RRWNet.

Automated Image Quality Gate (IQA): Classical computer vision checks for sharpness, lighting, and field-of-view before AI inference to eliminate false predictions on low-quality photos.

Test-Time Augmentation (TTA) & Uncertainty Estimation: Evaluates predictions across 3 augmented passes (baseline, horizontal flip, gamma adjustment) to reduce variance and calculate Shannon entropy for flagging borderline cases.

Explainable AI (XAI): Generates Grad-CAM visual heatmaps alongside a quantitative Attention Index to verify that the model focuses on true anatomical lesions.

Longitudinal Progression Tracking: Aligns historical patient scans to track microvascular narrowing and disease shift over time.

100% Offline Point-of-Care Deployment: Built with Streamlit and SQLite for instant local execution without external cloud dependencies or internet access.

🏗️ System Pipeline

                               ┌── IQA Gate (Pass/Fail)
                               │
Raw Fundus Photo ──► Quality Check ──┼── DR Engine (EfficientNetB3) ──► TTA Consensus & Entropy ──┐
                               │                                                                  ├──► PDF Diagnostic Report
                               └── HR Engine (RRWNet) ──────────► Zone B AVR Calculation ────────┤    & Streamlit Dashboard
                                                                                                  │
                                                                  Grad-CAM & Attention Index ─────┘


🛠️ Tech Stack

Frontend / UI: Streamlit

Deep Learning Framework: PyTorch, Torchvision

Computer Vision: OpenCV, PIL, Scikit-Image

Models: EfficientNetB3 (DR Classification), RRWNet (Vessel Segmentation)

Database & Storage: SQLite, ReportLab (PDF Generation)

Data & Analytics: NumPy, Pandas, Matplotlib

🚀 Quick Start & Installation

1. Clone the Repository

git clone https://github.com/your-username/retiscan-pro.git
cd retiscan-pro


2. Create a Virtual Environment & Install Dependencies

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt


3. Run the Streamlit App

streamlit run app.py


📄 Output & Diagnostic Reports

Every processed scan generates:

Interactive Dashboard View: Instant stage prediction, confidence scores, AVR ratios, and Grad-CAM overlays.

Downloadable Clinical PDF: Single-page summary complete with patient history, risk flags, attention scores, and specialist review prompts.

📜 License

Distributed under the MIT License. See LICENSE for more information.
