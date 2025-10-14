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

# ---------------------- 모델 로딩 -----------------------
print("Loading ONNX face model...")
onnx_session = ort.InferenceSession("emotion_model.onnx", providers=["CPUExecutionProvider"])
onnx_input_name = onnx_session.get_inputs()[0].name

print("Loading Keras audio model...")
audio_model = tf.keras.models.load_model("audio_emotion_model.keras")

# ---------------------- 클래스 정의 ----------------------
expression_labels = ["Angry", "Sad", "Frenzy", "Tear", "Happy", "Neutral"]

# ---------------------- 유틸 함수 -----------------------

def preprocess_image(png_bytes):
    import cv2
    image = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # CHW
    image = np.expand_dims(image, axis=0)  # NCHW
    return image

def preprocess_audio(wav_bytes):
    wav, sr = sf.read(io.BytesIO(wav_bytes))
    if len(wav.shape) > 1:
        wav = np.mean(wav, axis=1)  # stereo to mono
    wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)

    # Extract MFCCs (commonly used for speech emotion recognition)
    mfcc = librosa.feature.mfcc(y=wav, sr=16000, n_mfcc=40)  # shape: (40, time)
    mfcc_mean = np.mean(mfcc, axis=1)  # shape: (40,)

    # Reshape to match expected input shape: (1, 40, 1)
    mfcc_input = np.expand_dims(mfcc_mean, axis=(0, 2)).astype(np.float32)  # (1, 40, 1)
    return mfcc_input

# ---------------------- API -----------------------

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files or 'audio' not in request.files:
        return jsonify({'error': 'image and audio are required'}), 400

    try:
        # 이미지 처리
        img_bytes = request.files['image'].read()
        image_input = preprocess_image(img_bytes)
        onnx_output = onnx_session.run(None, {onnx_input_name: image_input})[0]
        face_pred = F.softmax(torch.tensor(onnx_output[0]), dim=-1)
        face_label = expression_labels[face_pred.argmax().item()]
        print(f"Face emotion: {face_label}")

        # 오디오 처리
        wav_bytes = request.files['audio'].read()
        audio_input = preprocess_audio(wav_bytes)
        audio_pred = audio_model.predict(audio_input)[0]
        audio_label = expression_labels[np.argmax(audio_pred)]
        print(f"Audio emotion: {audio_label}")

        return jsonify({
            'face': face_label,
            'audio': audio_label
        })

    except Exception as e:
        print("Error during inference:", str(e))
        return jsonify({'error': str(e)}), 500

# ---------------------- 시작 -----------------------

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
