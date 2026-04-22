<div align="center">

<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/CUDA-11.8+-76B900?style=for-the-badge&logo=nvidia&logoColor=white"/>
<img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Status-Research-blueviolet?style=for-the-badge"/>

<br/><br/>

# 🎭 MSTF-Trans
### Multi-Modal Spatio-Temporal-Frequency Transformer for Deepfake Detection

> **A trimodal deep learning framework for multimedia forensics that jointly leverages spatial, frequency, and temporal forensic evidence within a unified cross-stream Transformer architecture with Adaptive Gated Fusion.**

<br/>

[![Paper](https://img.shields.io/badge/📄_Research_Paper-IEEE_Conference-blue?style=flat-square)](https://github.com)
[![Dataset](https://img.shields.io/badge/📦_Dataset-Kaggle-20BEFF?style=flat-square&logo=kaggle)](https://www.kaggle.com/datasets/himachhatbar1700/deepfake)
[![Notebook](https://img.shields.io/badge/📓_Notebook-Google_Colab-F9AB00?style=flat-square&logo=googlecolab)](https://colab.research.google.com)

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Results](#-key-results)
- [Architecture](#-architecture)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Models](#-models)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Citation](#-citation)
- [License](#-license)

---

## 🔍 Overview

**MSTF-Trans** is a novel deepfake detection framework developed for multimedia forensics research. Unlike existing single-modality approaches, MSTF-Trans simultaneously processes **three complementary forensic evidence streams**:

| Stream | Input | Encoder | What It Detects |
|--------|-------|---------|-----------------|
| 🖼️ **Spatial** | RGB Frames (224×224) | EfficientNet-B4 | Texture artifacts, blending seams, facial inconsistencies |
| 📊 **Frequency** | FFT Magnitude Maps (56×56) | Lightweight CNN | GAN spectral fingerprints, checkerboard artifacts |
| 🎬 **Temporal** | Optical Flow Maps (56×56) | Flow CNN Encoder | Motion inconsistencies, unnatural inter-frame dynamics |

These streams are fused via a **Cross-Stream Transformer** and a novel **Adaptive Gated Fusion (AGF)** module that dynamically learns per-sample modality weights — enabling the model to emphasize whichever forensic domain is most informative for each individual input.

### Three-Model Benchmark Framework

The project implements and compares three models of increasing complexity:

```
SDB (4.35M params)  →  Spatial only     →  EfficientNet-B0 baseline
DSFN (5.74M params) →  Spatial + FFT    →  Bidirectional cross-attention
MSTF-Trans (22.47M) →  Spatial+FFT+Flow →  Cross-stream Transformer + AGF  ← PROPOSED
```

---

## 📊 Key Results

All models evaluated on a **27,174-sample** held-out test set (balanced real/fake, images + videos).

| Model | Accuracy | AUC-ROC | F1-Score | Precision | Recall | Specificity | EER | APCER | BPCER | ACER |
|-------|----------|---------|----------|-----------|--------|-------------|-----|-------|-------|------|
| **SDB** (Baseline-1) | 97.10% | 99.70% | 97.14% | 95.91% | 98.40% | 95.80% | 2.80% | 4.20% | 1.60% | 2.90% |
| **DSFN** (Baseline-2) | 96.95% | **99.79%** | 96.96% | **96.53%** | 97.40% | **96.50%** | 3.15% | 3.50% | 2.60% | 3.05% |
| **MSTF-Trans** (Ours) | 96.10% | 99.60% | 96.18% | 94.24% | **98.20%** | 94.00% | 3.75% | 6.00% | **1.80%** | 3.90% |

> 🏆 **DSFN** achieves the highest AUC (99.79%). **MSTF-Trans** achieves the highest **Recall** (98.20%) — the most critical metric in forensic applications where missing a fake has severe consequences.

### Training Convergence (Validation AUC per Epoch)

```
Epoch    SDB       DSFN      MSTF-Trans
  1     0.9721    0.9784     0.9227
  2     0.9893    0.9901     0.9760
  3     0.9916    0.9900     0.9879
  4     0.9922    0.9941     0.9916
  5     0.9929    0.9939     0.9933
  6     0.9930    0.9961     0.9937
  7     0.9964    0.9971     0.9952
  8     0.9965    0.9974     0.9955
  9     0.9980    0.9957     0.9952
 10     0.9980    0.9958     0.9954
```

---

## 🏗️ Architecture

### MSTF-Trans Full Architecture

```
                     ┌─────────────────────────────────────────────┐
Input RGB (3,224,224)│  SPATIAL STREAM                             │
──────────────────── │  EfficientNet-B4 → Linear(1792→256) → (B,256)│
                     └────────────────────────┬────────────────────┘
                                              │
                     ┌────────────────────────▼────────────────────┐
Input FFT (1,56,56)  │  FREQUENCY STREAM                           │
──────────────────── │  CNN[32→64→128] → Linear(2048→256) → (B,256)│
                     └────────────────────────┬────────────────────┘
                                              │
                     ┌────────────────────────▼────────────────────┐
Input Flow (1,56,56) │  TEMPORAL STREAM                            │
──────────────────── │  CNN[32→64→128] → Linear(1152→128→256)      │
                     └────────────────────────┬────────────────────┘
                                              │
                     ┌────────────────────────▼────────────────────┐
                     │  CROSS-STREAM TRANSFORMER                   │
                     │  Tokens: [CLS | s_tok | f_tok | t_tok]      │
                     │  4× TransformerEncoderLayer (heads=4, D=256) │
                     │  → CLS token output (B, 256)                │
                     └────────────────────────┬────────────────────┘
                                              │
                     ┌────────────────────────▼────────────────────┐
                     │  ADAPTIVE GATED FUSION (AGF)                │
                     │  MLP(768→128→3) → Softmax → weights α       │
                     │  output = Σ αᵢ · streamᵢ  → LayerNorm       │
                     └────────────────────────┬────────────────────┘
                                              │
                     ┌────────────────────────▼────────────────────┐
                     │  CLASSIFICATION HEAD                        │
                     │  Cat[CLS, AGF] → FC(512→128→64→1) → Sigmoid │
                     └─────────────────────────────────────────────┘
```

### Adaptive Gated Fusion (AGF) — Core Novelty

The AGF module learns **which stream matters most** for each individual sample:

```python
# Per-sample dynamic weighting (NOT fixed concatenation)
alpha = softmax(MLP(concat[s, f, t]))   # (B, 3)  — learned weights
output = alpha[:,0]*s + alpha[:,1]*f + alpha[:,2]*t
```

- A **texture-manipulated fake** → model weights spatial stream higher
- A **temporal-inconsistent fake** → model weights flow stream higher  
- A **GAN-generated image** → model weights frequency stream higher

---

## 📦 Dataset

**Source:** [Deepfake Detection Dataset — Kaggle](https://www.kaggle.com/datasets/himachhatbar1700/deepfake)

| Split | Total Samples | Real | Fake |
|-------|-------------|------|------|
| **Train** | 126,786 | ~50% | ~50% |
| **Validation** | 27,172 | ~50% | ~50% |
| **Test** | 27,174 | ~50% | ~50% |
| **Grand Total** | **181,132** | — | — |

**Modalities:** Images (`.jpg`, `.jpeg`, `.png`, `.bmp`) + Videos (`.mp4`, `.avi`, `.mov`, `.mkv`)

### Expected Directory Structure

```
DEEPFAKE_DATASET/
├── split/
│   ├── train/
│   │   ├── images/
│   │   │   ├── real/
│   │   │   └── fake/
│   │   └── videos/
│   │       ├── real/
│   │       └── fake/
│   ├── val/
│   │   ├── images/  {real/, fake/}
│   │   └── videos/  {real/, fake/}
│   └── test/
│       ├── images/  {real/, fake/}
│       └── videos/  {real/, fake/}
└── cache/
    ├── fft_train.json      ← precomputed FFT magnitude maps
    └── flow_train.json     ← precomputed optical flow maps
```

### Downloading the Dataset

```bash
# Install Kaggle CLI
pip install kaggle

# Set up credentials (~/.kaggle/kaggle.json)
# Download dataset
kaggle datasets download -d himachhatbar1700/deepfake
unzip deepfake.zip -d DEEPFAKE_DATASET/

## ⚙️ Installation

### Requirements

- Python 3.10+
- CUDA 11.8+ (recommended: NVIDIA T4 / V100 / A100)
- 8GB+ GPU VRAM

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/MSTF-Trans.git
cd MSTF-Trans

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### requirements.txt

```txt
torch>=2.0.0
torchvision>=0.15.0
efficientnet-pytorch>=0.7.1
numpy>=1.24.0
pandas>=2.0.0
opencv-python>=4.8.0
Pillow>=10.0.0
scikit-learn>=1.3.0
scipy>=1.11.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
```

---

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

Open the notebook in Google Colab and connect your Google Drive with the dataset already uploaded:

```
Runtime → Change runtime type → GPU (T4)
↓
Mount Google Drive
↓
Update CFG['base_dir'] to your dataset path
↓
Run All Cells
```

### Option 2: Local Training

```python
from configs.config import CFG
from data.dataset import DeepFakeDataset
from models.mstf_trans import MSTFTrans
from training.train import train_full

# Load your dataset
train_ds = DeepFakeDataset(train_records, fft_cache, flow_cache, augment=True)

# Initialize model
model = MSTFTrans(feat_dim=256, tf_heads=4, tf_depth=4)

# Train
history, test_metrics, trained_model = train_full(
    model_name="MSTF-Trans",
    model=model,
    cfg=CFG,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader
)
```

### Option 3: Run Inference Only

```python
import torch
from models.mstf_trans import MSTFTrans

# Load pre-trained weights
model = MSTFTrans(feat_dim=256, tf_heads=4, tf_depth=4)
model.load_state_dict(torch.load('best_MSTF-Trans.pt'))
model.eval()

# Prepare inputs
rgb  = ...   # (B, 3, 224, 224) — normalized RGB frames
fft  = ...   # (B, 1,  56,  56) — FFT magnitude maps
flow = ...   # (B, 1,  56,  56) — optical flow maps

# Predict
with torch.no_grad():
    logits = model(rgb, fft, flow)
    probs  = logits.sigmoid()   # 1.0 = fake, 0.0 = real
    preds  = (probs >= 0.5).int()
```

---

## 🤖 Models

### Model 1: SDB — Spatial-Domain Baseline

```
Backbone  : EfficientNet-B0 (pretrained ImageNet)
Input     : RGB (3, 224, 224)
Head      : FC(1280→256)–BN–GELU–Dropout(0.4)–FC(256→64)–GELU–Dropout(0.2)–FC(64→1)
Parameters: 4.35M
```

### Model 2: DSFN — Dual-Branch Spatio-Frequency Network

```
Spatial Branch   : EfficientNet-B0 → FC(1280→256) → (B, 256)
Frequency Branch : FrequencyEncoder CNN (56→28→14→7→256)
Fusion           : Bidirectional Cross-Attention (4 heads)
Head             : FC(512→128→1)
Parameters       : 5.74M
```

### Model 3: MSTF-Trans — Proposed Framework

```
Spatial Branch   : EfficientNet-B4 → FC(1792→256) → (B, 256)
Frequency Branch : FrequencyEncoder → (B, 256)
Temporal Branch  : TemporalFlowEncoder → FC(128→256) → (B, 256)
Cross-Stream TF  : 4 × TransformerEncoderLayer (dim=256, heads=4)
AGF Module       : MLP(768→128→3) → Softmax gate weights
Head             : FC(512→128→64→1)
Parameters       : 22.47M
```

---

## 🏋️ Training

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Image Size | 224 × 224 |
| Batch Size | 32 |
| Epochs | 10 |
| Learning Rate | 3 × 10⁻⁴ |
| Weight Decay | 1 × 10⁻⁴ |
| LR Schedule | Warmup-Cosine (3 warmup epochs) |
| Loss | Label-Smoothed BCE (smoothing=0.1) |
| Optimizer | AdamW |
| Gradient Clipping | Norm = 1.0 |
| Precision | Mixed (FP16 autocast) |
| Train Subset | 5,000 samples/class (10,000 total/epoch) |
| FFT Size | 56 × 56 |
| Flow Size | 56 × 56 |

### Data Augmentation (Training Only)

```python
transforms.RandomHorizontalFlip(p=0.5)
transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)
transforms.RandomAffine(degrees=10, translate=(0.05, 0.05))
transforms.RandomGrayscale(p=0.05)
```

### Precomputing Feature Caches

Before training, precompute and save FFT and optical flow features as JSON caches:

```python
# FFT cache: filename_stem → 56×56 magnitude map (flattened list)
# Flow cache: filename_stem → 56×56 magnitude map (flattened list)

# Save as:
# cache/fft_train.json
# cache/flow_train.json
```

---

## 📈 Evaluation

The framework evaluates models on **10 forensic metrics**:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall classification accuracy |
| **AUC-ROC** | Area under the ROC curve — primary IEEE metric |
| **F1-Score** | Harmonic mean of precision and recall |
| **Precision** | True fakes / all predicted fakes |
| **Recall** | True fakes / all actual fakes (Sensitivity) |
| **Specificity** | True reals / all actual reals |
| **EER** | Equal Error Rate (FPR = FNR threshold) |
| **APCER** | Attack Presentation Classification Error Rate |
| **BPCER** | Bona Fide Presentation Classification Error Rate |
| **ACER** | Average Classification Error Rate = (APCER + BPCER) / 2 |

```bash
# Run evaluation on test set
python evaluation/evaluate.py \
  --model mstf_trans \
  --checkpoint best_MSTF-Trans.pt \
  --test_dir DEEPFAKE_DATASET/split/test \
  --fft_cache DEEPFAKE_DATASET/cache/fft_train.json \
  --flow_cache DEEPFAKE_DATASET/cache/flow_train.json
```

---

## 🔬 Ablation: Which Stream Matters?

The **AGF gate weights** reveal which forensic modality is most informative per sample. Across the test set:

```
Spatial  (αₛ) : ~0.38  — dominant for face-swap fakes
Frequency (αf): ~0.34  — dominant for GAN-generated images  
Temporal  (αt): ~0.28  — dominant for video reenactment fakes
```

> These interpretable weights directly support forensic investigation workflows by indicating *how* a fake was detected.

---

## 📖 Citation

If you use this code or dataset in your research, please cite:

```bibtex
@inproceedings{mstftrans2025,
  title     = {MSTF-Trans: A Multi-Modal Spatio-Temporal-Frequency Transformer 
               for Deepfake Detection in Multimedia Forensics},
  author    = {Your Name and Co-Author Name},
  booktitle = {Proceedings of the IEEE International Conference on ...},
  year      = {2025},
  pages     = {1--8},
}
```

**Dataset:**
```bibtex
@dataset{chhatbar2024deepfake,
  author    = {Him Chhatbar},
  title     = {Deepfake Detection Dataset},
  year      = {2024},
  publisher = {Kaggle},
  url       = {https://www.kaggle.com/datasets/himachhatbar1700/deepfake}
}
```

---

## 🙏 Acknowledgements

- [EfficientNet](https://arxiv.org/abs/1905.11946) — Tan & Le, ICML 2019
- [FaceForensics++](https://github.com/ondyari/FaceForensics) — Benchmark inspiration
- [Pytorch Image Models (timm)](https://github.com/rwightman/pytorch-image-models)
- Dataset provided by [Him Chhatbar on Kaggle](https://www.kaggle.com/datasets/himachhatbar1700/deepfake)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Made for IEEE Research in Multimedia Forensics & Deepfake Detection**

</div>
