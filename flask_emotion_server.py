# ============================================================
# 🧠 Multimodal Emotion Recognition Flask Server
# 멀티모달 감정 인식 Flask 서버
# ============================================================
# This Flask API receives both video frame (image) and audio data,
# analyzes them using two pre-trained models (ONNX + Keras),
# and returns the detected emotions in JSON format.
#
# 이 Flask API는 영상 프레임(이미지)과 오디오 데이터를 동시에 입력받아,
# 사전에 학습된 두 모델(ONNX + Keras)을 이용해 감정을 분석하고
# JSON 형태로 결과를 반환합니다.
# ============================================================


from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import torch
import torch.nn.functional as F
import librosa
import soundfile as sf
import onnxruntime as ort
import tensorflow as tf
import io

app = Flask(__name__)
CORS(app)

# ------------------------------------------------------------
# [1] Model Loading / 모델 로딩
# ------------------------------------------------------------
# - Loads two pretrained models:
#   1) ONNX model (emotion_model.onnx): Facial emotion recognition
#   2) Keras model (audio_emotion_model.keras): Speech emotion recognition
#
# 두 개의 사전 학습 모델을 로드합니다:
#   1) ONNX 모델: 얼굴 감정 인식용
#   2) Keras 모델: 음성 감정 인식용
# ------------------------------------------------------------

print("Loading ONNX face model...")
onnx_session = ort.InferenceSession("emotion_model.onnx", providers=["CPUExecutionProvider"])
onnx_input_name = onnx_session.get_inputs()[0].name

print("Loading Keras audio model...")
audio_model = tf.keras.models.load_model("audio_emotion_model.keras")

# ------------------------------------------------------------
# [2] Emotion Labels / 감정 클래스 정의
# ------------------------------------------------------------
# Defines emotion categories used by both models.
# 두 모델에서 공통으로 사용하는 감정 클래스들을 정의합니다.
# ------------------------------------------------------------
expression_labels = ["Angry", "Sad", "Frenzy", "Tear", "Happy", "Neutral"]

# ------------------------------------------------------------
# [3] Preprocessing Functions / 전처리 함수
# ------------------------------------------------------------
# Includes two functions:
#   - preprocess_image(): decodes and normalizes image input
#   - preprocess_audio(): extracts MFCC features from audio
#
# 이미지와 오디오 데이터를 각각 모델 입력 형태에 맞게 변환하는 함수입니다.
# ------------------------------------------------------------

def preprocess_image(png_bytes):
    import cv2
    image = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # CHW
    image = np.expand_dims(image, axis=0)   # NCHW
    return image


def preprocess_audio(wav_bytes):
    wav, sr = sf.read(io.BytesIO(wav_bytes))
    if len(wav.shape) > 1:
        wav = np.mean(wav, axis=1)  # stereo → mono
    wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)

    # Extract MFCCs and reshape
    mfcc = librosa.feature.mfcc(y=wav, sr=16000, n_mfcc=40)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_input = np.expand_dims(mfcc_mean, axis=(0, 2)).astype(np.float32)
    return mfcc_input


# ------------------------------------------------------------
# [4] API Endpoint: /analyze
# ------------------------------------------------------------
# - Receives 'image' and 'audio' via POST request
# - Predicts emotion from both modalities
# - Returns JSON response with:
#     { "face": <label>, "audio": <label> }
#
# 클라이언트로부터 image와 audio 파일을 동시에 받아,
# 각각 얼굴 감정과 음성 감정을 예측하고 JSON으로 반환합니다.
# ------------------------------------------------------------

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files or 'audio' not in request.files:
        return jsonify({'error': 'image and audio are required'}), 400

    try:
        # ----- Face emotion (image) -----
        img_bytes = request.files['image'].read()
        image_input = preprocess_image(img_bytes)
        onnx_output = onnx_session.run(None, {onnx_input_name: image_input})[0]
        face_pred = F.softmax(torch.tensor(onnx_output[0]), dim=-1)
        face_label = expression_labels[face_pred.argmax().item()]
        print(f"Face emotion: {face_label}")

        # ----- Audio emotion (sound) -----
        wav_bytes = request.files['audio'].read()
        audio_input = preprocess_audio(wav_bytes)
        audio_pred = audio_model.predict(audio_input)[0]
        audio_label = expression_labels[np.argmax(audio_pred)]
        print(f"Audio emotion: {audio_label}")

        # JSON response
        return jsonify({
            'face': face_label,
            'audio': audio_label
        })

    except Exception as e:
        print("Error during inference:", str(e))
        return jsonify({'error': str(e)}), 500


# ------------------------------------------------------------
# [5] Run Flask Server / 서버 실행
# ------------------------------------------------------------
# Runs the Flask app on port 5000 and listens to all IP addresses.
# Flask 서버를 포트 5000에서 실행하며, 모든 네트워크 인터페이스 요청을 허용합니다.
# ------------------------------------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
