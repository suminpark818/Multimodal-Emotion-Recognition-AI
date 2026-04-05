# Multimodal Emotion Recognition System  
### (멀티모달 감정 인식 시스템)

This project implements an **AI-based emotion recognition pipeline** that analyzes both **facial expressions (image)** and **voice tone (audio)** in real time.  
이 프로젝트는 영상(얼굴 표정)과 음성(말투)을 동시에 분석해 감정을 실시간으로 인식하는 AI 기반 시스템입니다.

---

##  System Architecture / 시스템 구조

```

[Dataset Stage]
├── Audio (TESS, RAVDESS)
├── Image (AffectNet, custom dataset)

[Model Training]
├── Audio → CNN (MFCC features)
└── Image → ResNet18 → EfficientNet-B0 (Knowledge Distillation)

[Server Stage]
└── Flask REST API
├── emotion_model.onnx  (Face)
├── audio_emotion_model.keras  (Voice)
└── /analyze endpoint (returns JSON)

````

---

## 1. Audio Emotion Recognition  
### (오디오 감정 인식 모델)

### Overview  
- Dataset: **TESS** + **RAVDESS**  
- Features: **MFCC (Mel-Frequency Cepstral Coefficients)**  
- Model: **1D-CNN** built with TensorFlow / Keras  
- Emotions: `angry`, `happy`, `neutral`, `sad`

### Workflow  
1. Load `.wav` files from both datasets.  
2. Extract **100-dimensional MFCCs** and compute their mean.  
3. Normalize data using `StandardScaler`.  
4. Train a CNN with Conv1D → MaxPooling → Dense layers.  
5. Evaluate and save as `audio_emotion_model.keras`.

### 주요 특징  
- 감정별 데이터 균형 조정  
- 실시간 음성 입력 예측 지원  
- 학습 후 스케일러(`scaler.save`)와 모델을 함께 저장하여 Flask 서버에서 재사용  

---

## 2. Facial Emotion Recognition  
### (영상 감정 인식 모델)

### Overview  
- Base model: **ResNet18 (Teacher)**  
- Student model: **EfficientNet-B0 (KD)**  
- Datasets: AffectNet + Custom datasets  
- Emotions: `angry`, `sad`, `neutral`, `happy`

### Workflow  
1. Train **ResNet18 Teacher** with `AdamW + CosineAnnealingLR + EMA + Mixup`.  
2. Distill knowledge to **EfficientNet-B0 Student**:  
   \[
   L = \alpha L_{KD} + (1 - \alpha) L_{CE}
   \]
3. Export as **ONNX model** (`emotion_model.onnx`).  

### 특징  
- Knowledge Distillation으로 **경량화 + 고성능 동시 달성**  
- MPS / CPU / GPU 호환 (macOS M1~Windows)  
- 최적의 Macro-F1 기준 자동 저장  

---

## 3. Real-Time Multimodal Flask Server  
### (실시간 멀티모달 감정 분석 서버)

### Overview  
The Flask API receives both an **image frame** and an **audio sample**,  
processes them through their respective models,  
and returns the detected emotions as JSON.  

Flask 기반 REST API로, 클라이언트로부터 **이미지(얼굴)**와 **오디오(음성)**를 함께 받아  
두 모델을 병렬로 실행한 뒤 감정 결과를 JSON으로 반환합니다.

---

### API Endpoint  
`POST /analyze`

**Form-data Parameters:**
| Key | Type | Description |
|-----|------|-------------|
| `image` | PNG / JPEG | Face frame |
| `audio` | WAV | Voice sample |

**Example Response:**
```json
{
  "face": "Happy",
  "audio": "Neutral"
}
````

---

### Features

| Feature           | Description                           |
| ----------------- | ------------------------------------- |
| Dual Inference    | Face (ONNX) + Voice (Keras)           |
| Lightweight       | EfficientNet-B0 student model         |
| Cross-Platform    | Supports MPS, CPU, CUDA               |
| Integration Ready | Unity / OBS / Web compatible REST API |

---

## Emotion Classes / 감정 클래스

| Class         | Description     |
| ------------- | --------------- |
| Angry         | 분노              |
| Sad           | 슬픔              |
| Neutral       | 중립              |
| Happy         | 행복              |
| Frenzy / Tear | 격앙 / 울음 (영상 전용) |

---

## How to Run / 실행 방법

### 1. Environment Setup

```bash
pip install flask flask-cors onnxruntime tensorflow librosa soundfile torch torchvision
```

### 2. Run Flask Server

```bash
python flask_emotion_server.py
```

### 3. Send Request Example

```bash
curl -X POST \
  -F "image=@frame.png" \
  -F "audio=@voice.wav" \
  http://127.0.0.1:5000/analyze
```

---

## Integration Example / 연동 예시

* Unity / OBS 플러그인에서 웹캠 & 마이크 입력을 전송 가능
* 응답값을 Live2D / VTuber 아바타 표정 제어에 즉시 반영
* 예: 실시간 방송 중 감정에 따라 캐릭터 표정 자동 변경

---

## Model Summary

| Model                       | Framework | File                    | Description                        |
| --------------------------- | --------- | ----------------------- | ---------------------------------- |
| `emotion_model.onnx`        | ONNX      | Facial Expression Model | EfficientNet-B0 (KD from ResNet18) |
| `audio_emotion_model.keras` | Keras     | Speech Emotion Model    | 1D-CNN trained on TESS/RAVDESS     |
| `scaler.save`               | Joblib    | Feature Normalizer      | StandardScaler for MFCC            |

---


## Author Note

Developed as part of a **real-time emotion-aware VTuber interaction system**,
integrating multimodal AI for expressive virtual avatars.

이 프로젝트는 실시간 감정 인식 기반 **VTuber 인터랙션 시스템**의 핵심 모듈로,
AI 기반의 **표정 + 음성 감정 동기화**를 구현하기 위해 제작되었습니다.

---


