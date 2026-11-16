# 🫁 Chest X-Ray Pneumonia Detection

> Chest X-ray classifier built on EfficientNetB4 transfer learning, with lung segmentation and Grad-CAM heatmaps that show what each prediction was based on. First version built in 7 days.

🔗 **[Live Demo](https://chest-xray-pneumonia-detection-bice.vercel.app)**

## 🎯 Results

All figures are measured on the 624-image Kaggle test set (234 normal, 390 pneumonia).

| Metric | v3, lung-masked (deployed) | v2, unmasked + TTA x10 |
|--------|----------------------------|------------------------|
| Test Accuracy | **90.54%** | 93.59% |
| ROC-AUC | **0.9668** | 0.9834 |
| Sensitivity (Recall) | **96.92%** | 94.36% |
| Specificity | **79.91%** | 92.31% |
| F1 Score | **0.9276** | 0.9485 |
| Threshold | 0.8357 | 0.7533 |

**The live demo runs v3.** The two columns are not a like-for-like comparison:

- **v2** scored higher, but Grad-CAM showed it was looking at ECG leads, "R" side markers and image corners instead of the lungs. Its checkpoint and threshold were also chosen using the test set, so the 93.59% figure is optimistic.
- **v3** passes every X-ray through a lung segmentation model and blacks out everything outside the lungs, before both training and prediction. Its checkpoint and threshold were chosen on a patient-level validation split, and the test set was used once, at the end.

v3 catches more pneumonia cases (higher sensitivity) and raises more false alarms on normal scans (lower specificity).

## 🔬 What is Grad-CAM?

Grad-CAM highlights the regions of the X-ray that drove the model's prediction.

- 🔴 Red/Yellow → high attention
- 🔵 Blue → low attention

Two details of this implementation:

- The heatmap explains the class the model actually predicted. For a NORMAL result it shows the evidence for normal, not the faint leftover evidence for pneumonia.
- Gradients are taken from the pre-sigmoid logit. Near scores of 0 or 1 the sigmoid saturates and its gradients become noise.

The heatmap is drawn over the full, unmasked X-ray, and the app has a slider to compare the two.

## 🏗️ Model Architecture

```
Input X-ray (grayscale)
↓
Lung segmentation (torchxrayvision PSPNet) → black out everything outside the lungs
↓
380×380, repeated to 3 channels
↓
EfficientNetB4 (pretrained on ImageNet, fine-tuned on X-rays)
↓
GlobalAveragePooling2D
↓
BatchNormalization
↓
Dense(256, ReLU) + Dropout(0.5)
↓
Dense(128, ReLU) + Dropout(0.3)
↓
Dense(1, Sigmoid) → Normal / Pneumonia
```

If segmentation finds too little lung, the image is left unmasked and the API reports `"lung_mask": false`.

### Why EfficientNetB4 over DenseNet121?

The first version of this project used DenseNet121. EfficientNetB4 replaced it because:

- **Compound scaling** scales width, depth and resolution together
- **Native 380×380 resolution** captures more detail than 224×224
- It performed better than DenseNet121 on this dataset

## 📊 Dataset

- **Source:** [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Total images:** 5,863 as listed on Kaggle
- **Training folder:** NORMAL 1,341 vs PNEUMONIA 3,875
- **Test folder:** NORMAL 234 vs PNEUMONIA 390
- **Challenge:** roughly 3x class imbalance, handled with class weights

## 🔭 Training (v3)

The full pipeline is in [`notebooks/retrain_v3.ipynb`](notebooks/retrain_v3.ipynb).

- **Split:** the official validation folder has only 16 images, so the train and validation folders are pooled and about 1/7 of *patients* are held out for validation. No patient appears in both.
- **Lung masking:** every image is masked once and cached, so the model cannot learn from markers, text or borders.
- **Augmentation:** small rotation, translation, zoom, contrast and brightness changes. No horizontal flip, because the heart sits on the left.
- **Phase 1:** train the head only (base frozen), `lr=1e-3`, up to 10 epochs, early stopping with patience 3 on validation AUC.
- **Phase 2:** fine-tune the top 100 layers of EfficientNetB4 (BatchNorm layers stay frozen), `lr=1e-5`, up to 20 epochs, early stopping with patience 5 on validation AUC.
- **Threshold:** chosen on the validation split, at the point that maximizes sensitivity + specificity.
- **Evaluation:** the test set is used exactly once, after everything else is fixed.

Test-time augmentation is not used by v3 or by the live API. It was part of the v2 evaluation only.

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Model training | TensorFlow, Keras, Google Colab (T4 GPU) |
| Base model | EfficientNetB4 (ImageNet pretrained) |
| Lung segmentation | torchxrayvision PSPNet (PyTorch) |
| Explainability | Grad-CAM |
| Backend API | Flask, Python, OpenCV |
| Frontend | React, Vite |
| Model hosting | Hugging Face Hub |
| Backend hosting | Hugging Face Spaces (Docker) |
| Frontend hosting | Vercel |

## 🔌 API

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Liveness check; returns the model name |
| `/predict` | POST | Multipart upload with a `file` field (PNG or JPG) |

`/predict` returns the label, confidence, raw score, threshold, whether lung masking succeeded, and the original image and Grad-CAM overlay as base64 PNGs.

The model loads on the first `/predict` call, not at startup, so the server binds its port immediately. The first request is slow for that reason.

## 🚀 Run Locally

### Prerequisites

- Python 3.11
- Node.js 20+

### Backend

```bash
cd backend
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
pip install -r requirements.txt
python app.py
# Runs on http://localhost:5000
```

`requirements.txt` includes `torchxrayvision`, which installs PyTorch alongside TensorFlow, so the first install is large.

### Frontend

```bash
cd frontend
npm install
npm run dev
# Runs on http://localhost:5173
```

The frontend calls the hosted API by default. To use a local backend, change `API_URL` at the top of `frontend/src/App.jsx`.

### Model weights

The model (`best_model_v3.keras`) and its threshold (`optimal_threshold_v3.json`) are downloaded from the [Hugging Face Hub](https://huggingface.co/vijayesh684/chest-xray-densenet121) the first time a prediction is requested. The repository name dates from the DenseNet121 version; it now holds the EfficientNetB4 models.

## 📁 Project Structure

```
chest-xray-pneumonia/
├── backend/
│   ├── app.py              # Flask REST API (lazy model loading)
│   ├── model_utils.py      # Preprocessing, inference and Grad-CAM
│   ├── lung_mask.py        # Lung segmentation and masking
│   └── requirements.txt    # Python dependencies
├── frontend/
│   ├── public/samples/     # Sample X-rays from the test set
│   └── src/
│       ├── App.jsx         # Main React component
│       └── index.css       # Styles
├── notebooks/
│   └── retrain_v3.ipynb    # v3 training and evaluation
├── vercel.json
└── README.md
```

## 🕓 Version History

| Version | Model | Notes |
|---------|-------|-------|
| v1 | DenseNet121 | First end-to-end build: model, Grad-CAM, Flask API and React frontend |
| v2 | EfficientNetB4 + TTA x10 | 93.59% test accuracy; later found to rely on shortcuts outside the lungs |
| v3 | EfficientNetB4, lung-masked | Deployed. Patient-level validation, test set used once |

## ⚠️ Disclaimer

For research and educational purposes only.
Not a substitute for professional medical diagnosis.

## 👨‍💻 Author

**Vijayesh Thiyagarajan**
B.Tech Robotics & AI, SASTRA University

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/vijayesh-thiyagarajan)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat&logo=github)](https://github.com/AlsoBaker)
