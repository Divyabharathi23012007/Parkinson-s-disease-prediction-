# 🧠 NeuroScan — Parkinson's Disease AI Platform

A complete, professional web application for Parkinson's Disease detection using deep learning voice analysis.

---

## 🌟 Features

### 🏠 Home Page
- Animated hero with live model statistics
- Confusion matrix, ROC curve, ML comparison chart
- Disease overview with key statistics

### 🧠 Prediction Page
- Full 22-feature voice input form (organized by category)
- One-click sample loading (Parkinson's / Healthy)
- Real-time animated confidence gauge
- Probability visualization with confidence bars
- PDF report generation

### 📊 Model Performance
- Accuracy, AUC, Precision, F1 Score cards
- ROC Curve with AUC annotation
- Interactive training & validation curves (accuracy + loss)
- ML vs DL comparison table

### 🔍 SHAP Analysis
- Feature importance bar chart
- SHAP summary plot (impact direction)
- Waterfall chart for single prediction explanation

### 📋 Prediction History
- All saved predictions with timestamps
- Search and filter (All / Parkinson's / Healthy)
- Export to CSV download

### ℹ️ About Page
- Disease overview
- Dataset details (UCI repository)
- Model architecture diagram (layer by layer)
- Tech stack badges

---

## 🚀 Quick Start

### 1. Setup

```bash
# Make sure your model files are in backend/
ls backend/
# parkinsons_model.keras  scaler.pkl  parkinsons.csv  app.py

# Install dependencies
pip install -r backend/requirements.txt
```

### 2. Start Backend

```bash
cd backend
python3 app.py
# → Flask API running on http://localhost:5000
```

### 3. Open Frontend

```bash
# Simply open in browser:
open frontend/index.html
# or
xdg-open frontend/index.html   # Linux
```

---

## 📁 Project Structure

```
parkinsons_app/
├── backend/
│   ├── app.py                    # Flask REST API
│   ├── parkinsons_model.keras    # Trained DNN model
│   ├── scaler.pkl                # StandardScaler
│   ├── parkinsons.csv            # Dataset
│   ├── precomputed_stats.json    # Model metrics cache
│   └── requirements.txt
├── frontend/
│   └── index.html                # Complete React SPA (no build needed!)
├── start.sh                      # One-click start script
└── README.md
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/stats` | Model accuracy, AUC, confusion matrix, ROC curve |
| POST | `/api/predict` | Run prediction with 22 voice features |
| GET | `/api/history` | All prediction history |
| GET | `/api/history/csv` | Download history as CSV |
| POST | `/api/report` | Generate PDF report |
| GET | `/api/dataset/stats` | Feature statistics from dataset |

---

## 🤖 Model Architecture

```
Input (22 features)
  ↓
Dense(128) + BatchNorm + Dropout(0.3)
  ↓
Dense(64) + BatchNorm + Dropout(0.2)
  ↓
Dense(32, ReLU)
  ↓
Dense(16, ReLU)
  ↓
Dense(1, Sigmoid) → Parkinson's Probability
```

**Training Config:** Adam optimizer, Binary Crossentropy loss, EarlyStopping (patience=15), ReduceLROnPlateau

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 94.87% |
| ROC AUC | 98.62% |
| Sensitivity | 93.3% |
| Specificity | 100% |

---

## 🎨 Tech Stack

- **Backend:** Python 3, Flask, TensorFlow/Keras, scikit-learn, SHAP, ReportLab
- **Frontend:** React 18 (CDN, no build step), Recharts, Vanilla CSS
- **Data:** UCI Parkinson's Dataset (Max Little, Oxford University, 2008)

---

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**. It is **not** a substitute for professional medical diagnosis. Always consult a qualified neurologist for clinical assessment.

---

*Built as a Deep Learning & AI academic project.*
