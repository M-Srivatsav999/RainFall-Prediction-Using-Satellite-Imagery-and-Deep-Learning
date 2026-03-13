<div align="center">

# 🛰️ Real-Time Rainfall Prediction System

### Deep Learning-Based Precipitation Forecasting Using Live Satellite Imagery

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![NASA GIBS](https://img.shields.io/badge/NASA-GIBS%20API-0B3D91?style=for-the-badge&logo=nasa)](https://earthdata.nasa.gov)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br/>

> **CNN + Bidirectional LSTM + Multi-Head Attention**
> Real NASA Satellite Imagery · Open-Meteo Weather API · 5-Horizon Forecast · MC-Dropout Uncertainty

<br/>

![Dashboard Preview](https://img.shields.io/badge/Dashboard-4%20Tabs-blueviolet?style=flat-square)
![Free APIs](https://img.shields.io/badge/APIs-100%25%20Free%20%7C%20No%20Key%20Required-success?style=flat-square)
![Model](https://img.shields.io/badge/Model-ResNet--4%20%2B%20BiLSTM%20%2B%20Attention-orange?style=flat-square)

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Key Features](#-key-features)
- [What Makes This Different](#-what-makes-this-different)
- [Model Architecture](#-model-architecture)
- [Project Structure](#-project-structure)
- [File-by-File Explanation](#-file-by-file-explanation)
- [APIs Used](#-apis-used-100-free--no-key-required)
- [Satellite Imagery Layers](#-satellite-imagery-layers)
- [Installation & Setup](#-installation--setup)
- [How to Use](#-how-to-use)
- [Dashboard Tabs](#-dashboard-tabs)
- [Training Details](#-training-details)
- [Results & Performance](#-results--performance)
- [Known Limitations](#-known-limitations--future-work)
- [Technologies Used](#-technologies-used)

---

## 🌍 Overview

This project is a complete end-to-end deep learning system that predicts rainfall at **5 forecast horizons (1h, 3h, 6h, 12h, 24h)** using real-time satellite imagery fetched directly from **NASA's Global Imagery Browse Services (GIBS)**. 

Unlike traditional approaches that rely solely on numerical weather models or weather station data, this system:

- 🛰️ Fetches **actual satellite cloud imagery** from NASA — no synthetic data at inference
- 🧠 Processes images through a **ResNet CNN + Bidirectional LSTM + Multi-Head Attention** pipeline
- 📊 Outputs **multi-horizon forecasts with confidence intervals** using Monte Carlo Dropout
- 🌦️ Fuses live meteorological data from the **Open-Meteo weather API**
- 💻 Presents everything in a **professional 4-tab Streamlit dashboard**

All data sources are **completely free** with no API keys required.

---

## 🚀 Live Demo

> Deploy your own instance in minutes — see [Installation & Setup](#-installation--setup)

```
streamlit run app.py → http://localhost:8501
```

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🛰️ **Real NASA Satellite Data** | Live MODIS, GOES-East, and GPM IMERG imagery via OGC WMS protocol |
| 🧠 **Deep Learning Pipeline** | ResNet-4 CNN → BiLSTM → Multi-Head Self-Attention |
| 📈 **5 Forecast Horizons** | Simultaneous prediction for 1h, 3h, 6h, 12h, and 24h |
| 🎯 **Uncertainty Quantification** | MC-Dropout (20 samples) generates confidence intervals per horizon |
| 🌧️ **Rain Severity Classification** | 5 classes: No Rain, Light, Moderate, Heavy, Extreme |
| 🌤️ **Weather Data Fusion** | Live temperature, humidity, wind speed from Open-Meteo API |
| 📷 **Dual Input Modes** | Upload your own images OR fetch live NASA satellite frames |
| 🗂️ **7 Satellite Layers** | True Color, IR Thermal, GPM Rain Rate, Water Vapor, and more |
| 📥 **Downloadable Reports** | Export predictions as JSON with full horizon breakdown |
| 🖥️ **Professional Dashboard** | 4-tab Streamlit UI with dark theme and interactive charts |

---

## 🆚 What Makes This Different

### Comparison with Existing Systems

| Aspect | Traditional / Baseline | **This Project** |
|---|---|---|
| **Data Source** | Synthetic random / weather stations only | ✅ Real NASA GIBS satellite imagery |
| **Image Resolution** | 64 × 64 (low detail) | ✅ 128 × 128 (2× spatial detail) |
| **CNN Architecture** | Flat 3-layer CNN, no skip connections | ✅ ResNet-4 with residual skip connections |
| **Temporal Model** | Unidirectional LSTM (32 units) | ✅ Bidirectional LSTM (128 units) + Attention |
| **Forecast Output** | Single next-step value only | ✅ 5 horizons simultaneously |
| **Uncertainty** | None — single point estimate | ✅ MC-Dropout confidence intervals |
| **Rain Classification** | Raw number only | ✅ 5-class severity with colour coding |
| **Weather Fusion** | None | ✅ Open-Meteo live API |
| **Training Quality** | Random noise (R² ≈ -0.000) | ✅ Physics-correlated (R² ~0.75+) |
| **Loss Function** | MSE only | ✅ Combined MSE + MAE (outlier robust) |
| **LR Scheduling** | Constant learning rate | ✅ CosineAnnealingLR + early stopping |
| **Satellite Layers** | None | ✅ 7 MODIS/GOES/GPM layers |
| **API Cost** | N/A | ✅ 100% free, no API key |
| **UI** | Basic upload + single number | ✅ 4-tab dashboard + JSON export |

### Why Not Just Use Google Weather or IMD?

Those services provide **general city-level forecasts** using massive government supercomputers (NWP models). This project is different:

- **Google/IMD** → tells you it will rain today for a city
- **This system** → reads actual current satellite cloud patterns and tells you *how much* rain, *for how long*, and *with what confidence* — following the same deep learning philosophy as Google DeepMind's **GraphCast** and **NowcastNet**, scoped for regional nowcasting

This is a **precipitation nowcasting** system — a complementary tool especially useful for data-sparse regions where NWP models lack ground observations.

---

## 🏗️ Model Architecture

```
Input: T × 3 × 128 × 128   (5 sequential satellite frames)
           │
           ▼  (applied per frame independently)
┌─────────────────────────────────────────────────────┐
│                ResNet-4 CNN Extractor                │
│                                                     │
│   Conv2d(3→32) + BatchNorm + ReLU                  │
│         │                                           │
│   ┌─────┴─────┐                                    │
│   │ ResBlock  │  Conv(32→32) + BN + ReLU + skip    │
│   └─────┬─────┘                                    │
│         ▼                                           │
│   Conv2d(32→64, stride=2) + BatchNorm + ReLU       │
│   ResBlock(64→64)                                  │
│         ▼                                           │
│   Conv2d(64→128, stride=2) + BatchNorm + ReLU      │
│   ResBlock(128→128)                                │
│         ▼                                           │
│   Conv2d(128→256, stride=2) + BatchNorm + ReLU     │
│   ResBlock(256→256)                                │
│         ▼                                           │
│   GlobalAvgPool → Linear(256→256) → LayerNorm      │
└─────────────────────────────────────────────────────┘
           │  shape: (Batch, T, 256)
           ▼
┌─────────────────────────────────────────────────────┐
│         Bidirectional LSTM  (2 layers, 128 units)    │
│         Forward + Backward → concatenated 256-dim   │
│         Output shape: (Batch, T, 256)               │
└─────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────┐
│      Multi-Head Self-Attention  (4 heads)            │
│      Learns which frames are most rain-relevant     │
│      Residual connection + LayerNorm                │
└─────────────────────────────────────────────────────┘
           │  take last timestep → (Batch, 256)
           │
     ┌─────┴────────────┐
     ▼                  ▼
Regression Head     Classifier Head
─────────────       ───────────────
Linear(256→128)     Linear(256→64)
ReLU + Dropout      ReLU + Dropout
Linear(128→5)       Linear(64→5)
Softplus            Softmax
     │                  │
     ▼                  ▼
1h/3h/6h/12h/24h    No Rain / Light /
rainfall (mm/h)     Moderate / Heavy /
+ MC-Dropout CI     Extreme
```

### Why Each Component?

| Component | Why Used |
|---|---|
| **ResNet Skip Connections** | Prevents vanishing gradients in 4-stage CNN; enables richer feature extraction |
| **Bidirectional LSTM** | Reads cloud sequence both forward and backward; detects building vs dissipating storms |
| **Multi-Head Attention** | Explicitly learns which historical frames matter most for each prediction |
| **MC-Dropout** | Keeps dropout ON at inference; 20 samples give mean + std = uncertainty quantification |
| **Softplus output** | Guarantees non-negative rainfall values (physics constraint) |
| **Combined MSE + MAE loss** | MAE term makes training robust to heavy-rain outliers |

---

## 📂 Project Structure

```
rainfall-prediction/
│
├── app.py                    # Streamlit dashboard (4 tabs)
├── config.py                 # Hyperparameters, API URLs, layer definitions
├── models.py                 # Full model architecture (ResNet + BiLSTM + Attention)
├── train.py                  # Training pipeline with evaluation
├── predict.py                # Inference with MC-Dropout uncertainty
├── data_preprocessing.py     # Data pipeline + Open-Meteo API
├── satellite_fetch.py        # NASA GIBS WMS real-time imagery fetcher
│
├── requirements.txt          # Python dependencies
├── cnn_lstm_model.pth        # Trained model weights (generated by train.py)
├── scaler.pkl                # Fitted MinMaxScaler (generated by train.py)
│
└── .streamlit/
    └── config.toml           # Streamlit config (theme + watcher fix)
```

---

## 📄 File-by-File Explanation

### `config.py`
Central configuration file for the entire project.
- Image size (`128×128`), sequence length (`5 frames`), forecast horizons (`[1,3,6,12,24]`)
- NASA GIBS WMS endpoint and all satellite layer identifiers
- Rain category boundaries and severity colour codes
- Device configuration (CPU/GPU auto-detection)

### `satellite_fetch.py`
Handles all real-time satellite data acquisition from NASA.
- `fetch_satellite_image()` — single WMS request for any region + date combination
- `fetch_sequence()` — fetches 5 consecutive daily frames for temporal context
- `LAYERS` dictionary — 7 imagery types with full WMS layer names
- `LOCATIONS` dictionary — pre-configured bounding boxes for India, Bay of Bengal, SE Asia, and more
- `extract_cloud_features()` — physics-based feature proxies (cloud cover, cold cloud fraction, moisture index)
- `get_available_dates()` — returns last 7 available dates accounting for GIBS 3-5h latency

### `data_preprocessing.py`
Generates physics-correlated training data and handles live weather API calls.
- `generate_sample_data()` — rainfall derived as `cloud_cover^1.5 × humidity × instability × 20`
- `fetch_weather_metadata()` — calls Open-Meteo API for current + 24h hourly forecast
- `SatelliteRainDataset` — PyTorch dataset with augmentation (horizontal flip, brightness jitter)
- Proper train/test split with `MinMaxScaler` fitted only on training data

### `models.py`
Complete deep learning architecture definition.
- `ResBlock` — residual skip connection block preventing vanishing gradients
- `DeepCNNExtractor` — 4-stage ResNet CNN with GlobalAvgPool and LayerNorm projection
- `TemporalAttention` — MultiheadAttention with residual connection and LayerNorm
- `RainfallNet` — full model combining CNN, BiLSTM, Attention, and dual output heads
- `predict_with_uncertainty()` — MC-Dropout inference returning mean + standard deviation

### `train.py`
Complete training pipeline with evaluation.
- `CombinedLoss` — `MSE + 0.5 × MAE` for outlier robustness
- `AdamW` optimizer with `CosineAnnealingLR` scheduler
- Early stopping with patience=8, saves best model checkpoint
- Per-horizon R² and MAE evaluation for all 5 forecast windows
- Saves training plots (loss curves, scatter plots, R² bar chart)

### `predict.py`
Inference engine with physics-based cloud signal integration.
- Model and scaler loaded once and cached globally for fast repeated predictions
- `_extract_cloud_signal()` — extracts 5 real cloud features from image pixels
- Physics signal anchors rainfall magnitude; model provides temporal decay ratios
- `_classify_rain()` — maps mm/h values to 5-class severity categories
- Full confidence scoring based on cloud signal clarity

### `app.py`
Complete Streamlit web dashboard.
- **Tab 1 (Predict)** — live satellite fetch or image upload → prediction → 5-horizon chart + download
- **Tab 2 (Satellite View)** — side-by-side multi-layer NASA imagery explorer
- **Tab 3 (Weather Data)** — Open-Meteo live conditions + 24h precipitation bar chart
- **Tab 4 (Model Info)** — comparison table, architecture diagram, feature explanations

---

## 🌐 APIs Used (100% Free | No Key Required)

### NASA GIBS — Global Imagery Browse Services

```
Endpoint: https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi
Protocol: OGC WMS 1.1.1 (standard HTTP GET)
Coverage: Global
Latency:  Daily products ~3-5 hours after observation
          GOES-East products every 10 minutes
```

Example request:
```
GET https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi
    ?SERVICE=WMS
    &REQUEST=GetMap
    &LAYERS=MODIS_Terra_CorrectedReflectance_TrueColor
    &BBOX=70,10,90,30
    &WIDTH=512&HEIGHT=512
    &TIME=2024-09-28
```

### Open-Meteo — Weather Forecast API

```
Endpoint: https://api.open-meteo.com/v1/forecast
Protocol: REST/JSON
Coverage: Global
Data:     Current conditions + 24h hourly forecast
```

Variables fetched: `precipitation`, `cloudcover`, `relativehumidity_2m`, `windspeed_10m`, `weathercode`

---

## 🛰️ Satellite Imagery Layers

| Layer | Source | Update Frequency | Rain Relevance |
|---|---|---|---|
| **True Color (MODIS Terra)** | Terra satellite | Daily | Cloud shape and coverage |
| **True Color (MODIS Aqua)** | Aqua satellite | Daily | Complementary daily coverage |
| **IR Thermal (GOES-East Band 13)** | GOES-East | 10 minutes | Cold cloud tops = heavy rain |
| **GPM Precipitation Rate (IMERG)** | GPM constellation | 30 minutes | Direct rain rate measurement |
| **Cloud Top Temperature** | MODIS | Daily | Deep convection indicator |
| **Water Vapor (GOES-East Band 9)** | GOES-East | 10 minutes | Moisture transport and jet streams |
| **Snow & Ice Cover** | MODIS | Daily | Snowmelt flood indicator |

---

## 🔧 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Internet connection (for NASA GIBS and Open-Meteo APIs)

### Step 1 — Clone the Repository

```bash
git clone https://github.com/your-username/rainfall-prediction-deep-learning.git
cd rainfall-prediction-deep-learning
```

### Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Train the Model

```bash
python train.py
```

> ⏱️ Takes approximately **2-3 minutes** on CPU.  
> Generates `cnn_lstm_model.pth` and `scaler.pkl` in the project folder.

### Step 4 — Launch the Dashboard

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 📖 How to Use

### Option A — Live NASA Satellite (Recommended)

1. Open the **Predict** tab
2. In the sidebar, select your **Region** (e.g. India - Hyderabad)
3. Set **Data Source** to `🛰️ Live NASA Satellite`
4. Click **"Fetch Live Satellite Data"** — 5 frames will load automatically
5. Click **"PREDICT RAINFALL"**
6. View 5-horizon forecast chart with uncertainty bands
7. Download results as JSON if needed

### Option B — Upload Your Own Images

1. Set **Data Source** to `📁 Upload Images`
2. Upload **2 or more** satellite cloud images (JPG/PNG)
3. Click **"PREDICT RAINFALL"**

> 💡 Best results with top-down satellite imagery from NASA Worldview (worldview.earthdata.nasa.gov) or Zoom Earth (zoom.earth)

---

## 🖥️ Dashboard Tabs

### Tab 1 — Predict
- Dual input mode: Live NASA satellite or manual image upload
- 5-horizon forecast bar chart with uncertainty error bars
- Rain severity badge with colour coding
- Confidence score display
- Downloadable JSON prediction report

### Tab 2 — Satellite View
- Date selector (last 7 days of available NASA imagery)
- Side-by-side comparison of multiple satellite layers
- Cloud feature metrics: cloud cover %, cold cloud fraction, moisture index
- Region selector with pre-configured bounding boxes

### Tab 3 — Weather Data
- Live current conditions: temperature, humidity, wind speed, cloud cover
- 24-hour precipitation bar chart
- Hourly precipitation probability curve
- Weather condition description

### Tab 4 — Model Info
- Full comparison table vs. baseline system
- Architecture diagram
- Per-horizon R² and MAE scores from training
- Feature importance explanation
- Known limitations and future work roadmap

---

## 🏋️ Training Details

| Parameter | Value |
|---|---|
| **Image Size** | 128 × 128 pixels |
| **Sequence Length** | 5 temporal frames |
| **CNN Stages** | 4 (32 → 64 → 128 → 256 filters) |
| **LSTM Units** | 128 (bidirectional → 256 effective) |
| **Attention Heads** | 4 |
| **Forecast Horizons** | 5 (1h, 3h, 6h, 12h, 24h) |
| **Loss Function** | MSE + 0.5 × MAE |
| **Optimizer** | AdamW (lr=1e-3, weight_decay=1e-4) |
| **LR Scheduler** | CosineAnnealingLR (T_max=50) |
| **Early Stopping** | patience=8, monitors validation loss |
| **MC-Dropout Samples** | 20 inference passes |
| **Training Data** | Physics-correlated synthetic sequences |
| **Augmentation** | Horizontal flip + brightness jitter |

### Training Data Generation

Rainfall is generated using physical relationships:

```python
rain = cloud_cover^1.5 × humidity × convective_instability × 20
```

This ensures the model learns a genuine signal between cloud patterns and precipitation rather than fitting random noise (which caused R² ≈ -0.000 in the baseline).

---

## 📊 Results & Performance

### Model Performance (Synthetic Test Set)

| Horizon | R² Score | MAE (mm/h) |
|---|---|---|
| 1 hour | ~0.78 | ~1.2 |
| 3 hours | ~0.75 | ~1.4 |
| 6 hours | ~0.71 | ~1.7 |
| 12 hours | ~0.65 | ~2.1 |
| 24 hours | ~0.58 | ~2.6 |

### Rain Category Mapping

| Category | Range | Color |
|---|---|---|
| No Rain | 0 – 1 mm/h | ⚪ Grey |
| Light Rain | 1 – 5 mm/h | 🟢 Green |
| Moderate Rain | 5 – 15 mm/h | 🟡 Yellow |
| Heavy Rain | 15 – 35 mm/h | 🟠 Orange |
| Extreme Rain | 35+ mm/h | 🔴 Red |

---

## ⚠️ Known Limitations & Future Work

### Current Limitations

- **Synthetic Training Data** — Model is trained on physics-correlated synthetic data, not real paired satellite+rainfall observations. Production accuracy would improve significantly with labelled real data (IMERG + MODIS paired datasets).

- **Daily Temporal Resolution** — Current pipeline uses daily satellite composites. Storm-scale nowcasting (15-min intervals) requires GOES-East 10-minute imagery integration.

- **Regional Scale Only** — Operates at ~100km regional scale. Storm-cell level prediction requires radar data (e.g., NOAA MRMS network).

- **No Ground Truth Validation** — Predictions are not validated against rain gauge networks or IMERG ground truth in the current version.

### Future Work Roadmap

- [ ] Fine-tune on real MODIS + GPM IMERG paired datasets (publicly available)
- [ ] Integrate GOES-East 10-minute imagery for true nowcasting
- [ ] Add radar data fusion (NOAA MRMS) for storm-cell resolution
- [ ] Implement transformer-based architecture (ViT) to replace CNN
- [ ] Add user location detection for automatic region selection
- [ ] Deploy on Streamlit Cloud with GPU acceleration

---

## 🛠️ Technologies Used

| Technology | Version | Purpose |
|---|---|---|
| **Python** | 3.8+ | Core language |
| **PyTorch** | 2.1.0 | Deep learning framework |
| **Streamlit** | 1.29.0 | Web dashboard |
| **NumPy** | 1.25.2 | Numerical computing |
| **Pillow** | 10.1.0 | Image processing |
| **scikit-learn** | 1.3.2 | Scaler + metrics |
| **Matplotlib** | 3.8.2 | Training plots |
| **Plotly** | 5.17.0 | Interactive charts |
| **Requests** | 2.31.0 | HTTP API calls |
| **joblib** | 1.3.2 | Model serialization |
| **NASA GIBS** | WMS 1.1.1 | Satellite imagery (free) |
| **Open-Meteo** | v1 | Weather forecast (free) |

---

## 📁 Requirements

```txt
torch==2.1.0
torchvision==0.16.0
scikit-learn==1.3.2
numpy==1.25.2
pandas==2.1.4
matplotlib==3.8.2
streamlit>=1.29.0
pillow>=10.0.0
plotly>=5.17.0
joblib>=1.3.2
requests>=2.31.0
```

---

## 🎓 Academic Context

This project was developed as a mini project for the **Department of Computer Science & Engineering**, demonstrating the application of deep learning to real-world meteorological problems.

The architecture follows the same fundamental approach as recent state-of-the-art systems:

| System | Organization | Approach |
|---|---|---|
| **GraphCast** | Google DeepMind | Graph Neural Network on ERA5 reanalysis |
| **NowcastNet** | Google Research | CNN + LSTM for precipitation nowcasting |
| **FourCastNet** | NVIDIA | Fourier Neural Operator on global atmosphere |
| **AIFS** | ECMWF | Graph Neural Network replacing traditional NWP |
| **This Project** | Mini Project | ResNet CNN + BiLSTM + Attention on NASA imagery |

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- **NASA Earthdata / GIBS** — for providing free open access to global satellite imagery
- **Open-Meteo** — for the free weather forecast API
- **Global Precipitation Measurement (GPM) Mission** — for IMERG precipitation data
- **PyTorch Team** — for the deep learning framework
- **Streamlit** — for the web application framework

---

<div align="center">

**Built with ❤️ using NASA Open Data**

⭐ Star this repository if you found it useful!

</div>
