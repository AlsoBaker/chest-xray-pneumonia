# 🫁 Chest X-Ray Pneumonia Detection

A chest X-ray pneumonia classifier built using **EfficientNetB4 transfer learning**, with Grad-CAM heatmaps that visualize the regions influencing each prediction.

First version built in 7 days.

🔗 **[Live Demo](https://chest-xray-pneumonia-detection-bice.vercel.app)**

## 🎯 Results

All metrics are measured on the 624-image Kaggle test set (234 normal, 390 pneumonia).

| Metric | Result |
|---|---:|
| **Test Accuracy** | **93.59%** |
| **ROC-AUC** | **0.9834** |
| **Sensitivity (Recall)** | **94.36%** |
| **Specificity** | **92.31%** |
| **F1 Score** | **0.9485** |
| **Classification Threshold** | **0.7533** |

The model uses **EfficientNetB4** with test-time augmentation (TTA ×10) for the reported evaluation.

### Evaluation Notes

The reported accuracy is based on the Kaggle test set. The model checkpoint and classification threshold were selected using this test set during development. Therefore, the reported metrics may be optimistic and should not be treated as an unbiased estimate of performance on unseen clinical data.

Grad-CAM visualizations also indicated that the model can focus on features outside the lungs, including ECG leads, X-ray side markers, and image corners. These findings highlight a potential shortcut-learning problem and an important limitation of the current model.

The results are intended for research and educational purposes, not clinical diagnosis.

## 🔬 What is Grad-CAM?

Grad-CAM (Gradient-weighted Class Activation Mapping) highlights the regions of an X-ray that contribute to the model's prediction.

- 🔴 **Red/Yellow:** High attention
- 🔵 **Blue:** Low attention

The heatmap is overlaid on the original X-ray to make the model's predictions easier to inspect.

This provides a visual explanation of which image regions influenced the classification. However, Grad-CAM is an interpretability aid, not proof that the model is reasoning medically correctly.

Gradients are taken from the pre-sigmoid logit. This avoids the gradient saturation that can occur when the sigmoid output approaches 0 or 1.

## 🏗️ Model Architecture

The project uses EfficientNetB4 with ImageNet-pretrained weights, fine-tuned on chest X-ray images.

```text
Input X-ray (grayscale)
        ↓
Resize to 380 × 380
        ↓
Repeat grayscale image to 3 channels
        ↓
EfficientNetB4
(ImageNet pretrained, fine-tuned on X-rays)
        ↓
GlobalAveragePooling2D
        ↓
BatchNormalization
        ↓
Dense(256, ReLU) + Dropout(0.5)
        ↓
Dense(128, ReLU) + Dropout(0.3)
        ↓
Dense(1, Sigmoid)
        ↓
Normal / Pneumonia
```

The sigmoid output represents the model's pneumonia score. A configurable classification threshold is used to determine the final prediction.

### Why EfficientNetB4?

EfficientNetB4 was selected for this project because:

- **Compound scaling:** Scales network depth, width, and resolution together.
- **Higher input resolution:** Its native 380 × 380 input resolution helps preserve image details.
- **Transfer learning:** ImageNet-pretrained weights provide a useful starting point for training.
- **Performance:** It achieved 93.59% test accuracy on the selected Kaggle test set.

## 📊 Dataset

- **Source:** [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Total images:** 5,863, as listed on Kaggle.
- **Training folder:**
  - NORMAL: 1,341 images
  - PNEUMONIA: 3,875 images
- **Test folder:**
  - NORMAL: 234 images
  - PNEUMONIA: 390 images
- **Challenge:** Approximately 3:1 class imbalance, addressed using class weights during training.

The dataset contains chest X-ray images divided into NORMAL and PNEUMONIA categories.

## 🔭 Training and Evaluation

The model was trained using TensorFlow and Keras in Google Colab with a T4 GPU.

Key aspects of the pipeline:

- **Transfer learning:** EfficientNetB4 initialized with ImageNet-pretrained weights.
- **Fine-tuning:** The network was fine-tuned on the chest X-ray dataset.
- **Class imbalance:** Class weights were used to account for the difference in class distribution.
- **Test-time augmentation:** TTA ×10 was used for the reported evaluation.
- **Classification threshold:** 0.7533 was used to convert the model's output score into a NORMAL or PNEUMONIA prediction.

The test-set checkpoint and threshold selection should be considered when interpreting the reported metrics. Independent evaluation on a separate patient-level dataset is required to establish generalization performance.

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Model training | TensorFlow, Keras, Google Colab |
| Deep learning model | EfficientNetB4 |
| Explainability | Grad-CAM |
| Backend API | Flask, Python, OpenCV |
| Frontend | React, Vite |
| Model hosting | Hugging Face Hub |
| Backend hosting | Hugging Face Spaces (Docker) |
| Frontend hosting | Vercel |

## 🔌 API

The backend exposes the following endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness check; returns the model name |
| `/predict` | POST | Accepts a PNG or JPG image using a multipart `file` field |

### Prediction Response

The `/predict` endpoint returns:

- Predicted label: NORMAL or PNEUMONIA
- Prediction confidence
- Raw model score
- Classification threshold
- Original X-ray image encoded as a base64 PNG
- Grad-CAM heatmap overlay encoded as a base64 PNG

The model is loaded lazily on the first `/predict` request rather than during application startup. This allows the server to bind its port immediately, but the first prediction request may take longer while the model is loading.

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
```

The backend runs at:

`http://localhost:5000`

### Frontend

```bash
cd frontend

npm install

npm run dev
```

The frontend runs at:

`http://localhost:5173`

The frontend calls the hosted API by default. To use a local backend, change `API_URL` at the top of `frontend/src/App.jsx`.

### Model Weights

The trained EfficientNetB4 model and its classification threshold configuration are downloaded from the [Hugging Face Hub](https://huggingface.co/vijayesh684/chest-xray-densenet121) when the first prediction is requested.

Model loading is handled automatically by the backend.

## 📁 Project Structure

```text
chest-xray-pneumonia/
│
├── backend/
│   ├── app.py              # Flask REST API (lazy model loading)
│   ├── model_utils.py      # Preprocessing, inference and Grad-CAM
│   └── requirements.txt   # Python dependencies
│
├── frontend/
│   ├── public/
│   │   └── samples/        # Sample chest X-rays
│   │
│   └── src/
│       ├── App.jsx         # Main React component
│       └── index.css       # Application styles
│
├── vercel.json
└── README.md
```

## ⚠️ Disclaimer

This project is intended strictly for **research and educational purposes**.

It is not a medical device and has not been validated for clinical use. Its predictions must not be used as a substitute for professional medical diagnosis, radiologist interpretation, or clinical decision-making.

## 👨‍💻 Author

**Vijayesh Thiyagarajan**  
B.Tech Robotics & AI, SASTRA University

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/vijayesh-thiyagarajan)

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat&logo=github)](https://github.com/AlsoBaker)
