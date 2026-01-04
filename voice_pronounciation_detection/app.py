from flask import Flask, request, jsonify
import os
import pickle
import numpy as np
import librosa
import pandas as pd # Although not directly used by predictor, it's good practice to include if the original classes used it.
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIG ====================
class Config:
    """Configuration parameters for the model"""
    SAMPLE_RATE = 22050  # Audio sample rate
    DURATION = 5  # Maximum audio duration in seconds
    N_MFCC = 13  # Number of MFCC coefficients
    N_CHROMA = 12  # Number of chroma features
    N_MEL = 128  # Number of mel bands
    N_CONTRAST = 7  # Number of spectral contrast bands

    # SVM parameters (not directly used by predictor, but good to keep consistent)
    SVM_KERNEL = 'rbf'
    SVM_C = 10.0
    SVM_GAMMA = 'scale'
    RANDOM_STATE = 42
    TEST_SIZE = 0.2

# ==================== FEATURE EXTRACTION ====================
class AudioFeatureExtractor:
    """Extract audio features from wav files"""

    def __init__(self, config=Config()):
        self.config = config

    def load_audio(self, file_path):
        """Load audio file with error handling"""
        try:
            y, sr = librosa.load(
                file_path,
                sr=self.config.SAMPLE_RATE,
                duration=self.config.DURATION
            )
            return y, sr
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, None

    def extract_mfcc(self, y, sr):
        """Extract MFCC features"""
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=self.config.N_MFCC
        )
        # Statistical features: mean, std, min, max
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        mfcc_min = np.min(mfcc, axis=1)
        mfcc_max = np.max(mfcc, axis=1)
        return np.concatenate([mfcc_mean, mfcc_std, mfcc_min, mfcc_max])

    def extract_chroma(self, y, sr):
        """Extract chroma features"""
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)
        return np.concatenate([chroma_mean, chroma_std])

    def extract_mel_spectrogram(self, y, sr):
        """Extract mel-spectrogram features"""
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=self.config.N_MEL
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_mean = np.mean(mel_db, axis=1)
        mel_std = np.std(mel_db, axis=1)
        return np.concatenate([mel_mean, mel_std])

    def extract_spectral_contrast(self, y, sr):
        """Extract spectral contrast features"""
        contrast = librosa.feature.spectral_contrast(
            y=y,
            sr=sr,
            n_bands=self.config.N_CONTRAST - 1
        )
        contrast_mean = np.mean(contrast, axis=1)
        contrast_std = np.std(contrast, axis=1)
        return np.concatenate([contrast_mean, contrast_std])

    def extract_zero_crossing_rate(self, y):
        """Extract zero crossing rate"""
        zcr = librosa.feature.zero_crossing_rate(y)
        return np.array([np.mean(zcr), np.std(zcr)])

    def extract_spectral_rolloff(self, y, sr):
        """Extract spectral rolloff"""
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        return np.array([np.mean(rolloff), np.std(rolloff)])

    def extract_rms_energy(self, y):
        """Extract RMS energy"""
        rms = librosa.feature.rms(y=y)
        return np.array([np.mean(rms), np.std(rms)])

    def extract_all_features(self, file_path):
        """Extract all features from audio file"""
        y, sr = self.load_audio(file_path)

        if y is None:
            return None

        # Extract all features
        mfcc_features = self.extract_mfcc(y, sr)
        chroma_features = self.extract_chroma(y, sr)
        mel_features = self.extract_mel_spectrogram(y, sr)
        contrast_features = self.extract_spectral_contrast(y, sr)
        zcr_features = self.extract_zero_crossing_rate(y)
        rolloff_features = self.extract_spectral_rolloff(y, sr)
        rms_features = self.extract_rms_energy(y)

        # Combine all features
        feature_vector = np.concatenate([
            mfcc_features,      # 52 features (13*4)
            chroma_features,    # 24 features (12*2)
            mel_features,       # 256 features (128*2)
            contrast_features,  # 14 features (7*2)
            zcr_features,       # 2 features
            rolloff_features,   # 2 features
            rms_features        # 2 features
        ])

        return feature_vector

# ==================== PREDICTOR ====================
class PronunciationPredictor:
    """Make predictions on new audio"""

    def __init__(self, models, feature_extractor):
        self.models = models
        self.feature_extractor = feature_extractor

    def predict(self, audio_path, target_word):
        """Predict pronunciation quality"""
        target_word = target_word.lower().strip()

        # Check if model exists
        if target_word not in self.models:
            available_words = ', '.join(self.models.keys())
            return {
                'success': False,
                'error': f"No model found for word '{target_word}'",
                'available_words': available_words
            }

        # Extract features
        features = self.feature_extractor.extract_all_features(audio_path)

        if features is None:
            return {
                'success': False,
                'error': 'Failed to extract features from audio file'
            }

        # Get model components
        model_data = self.models[target_word]
        model = model_data['model']
        scaler = model_data['scaler']
        encoder = model_data['encoder']

        # Scale features
        features_scaled = scaler.transform(features.reshape(1, -1))

        # Predict
        prediction = model.predict(features_scaled)[0]
        prediction_proba = model.predict_proba(features_scaled)[0]

        # Get result
        result = encoder.inverse_transform([prediction])[0]
        confidence = prediction_proba[prediction]

        return {
            'success': True,
            'word': target_word,
            'prediction': result,
            'confidence': confidence,
            'probabilities': {
                'Correct': prediction_proba[0] if encoder.classes_[0] == 'Correct' else prediction_proba[1],
                'Incorrect': prediction_proba[1] if encoder.classes_[0] == 'Correct' else prediction_proba[0]
            },
            'model_accuracy': model_data['accuracy'],
            'cv_accuracy': model_data['cv_scores'].mean()
        }


app = Flask(__name__)

# Global variables for model and feature extractor
predictor = None

# Load the models and feature extractor at application startup
def load_predictor():
    global predictor
    # IMPORTANT: Update this path if your model file is in a different location in the deployed environment
    MODEL_PATH = './pronunciation_models.pkl' 

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        # Attempt to load from current directory as fallback for local testing
        MODEL_PATH = 'pronunciation_models.pkl'
        if not os.path.exists(MODEL_PATH):
            print(f"Error: Model file not found in current directory either: {MODEL_PATH}")
            return

    try:
        with open(MODEL_PATH, 'rb') as f:
            models = pickle.load(f)
        print(f"✓ Models loaded from: {MODEL_PATH}")

        config = Config()
        feature_extractor = AudioFeatureExtractor(config)
        predictor = PronunciationPredictor(models, feature_extractor)
        print("✓ Pronunciation Predictor initialized.")
    except Exception as e:
        print(f"Error loading predictor: {e}")

@app.before_request
def initialize_predictor():
    load_predictor()

@app.route('/', methods=['GET'])
def home():
    return "Welcome to the Pronunciation Assessment API!"

@app.route('/predict', methods=['POST'])
def predict_pronunciation():
    if predictor is None:
        return jsonify({"success": False, "error": "Model not loaded. Please ensure models are available and restart the server."}), 500

    # Expecting form data with 'audio_file' and 'target_word'
    if 'audio_file' not in request.files or 'target_word' not in request.form:
        return jsonify({"success": False, "error": "Missing 'audio_file' or 'target_word' in request."}), 400

    audio_file = request.files['audio_file']
    target_word = request.form['target_word']

    # Save the audio file temporarily. Ensure /tmp exists or choose another writable location.
    temp_audio_path = os.path.join('./tmp/', audio_file.filename)
    audio_file.save(temp_audio_path)

    prediction_result = predictor.predict(temp_audio_path, target_word)

    # Clean up temporary file
    os.remove(temp_audio_path)

    if prediction_result['success']:
        return jsonify(prediction_result)
    else:
        # Return 400 for client-side errors like missing model for word or feature extraction failure
        status_code = 400 if 'No model found' in prediction_result.get('error', '') or 'Failed to extract features' in prediction_result.get('error', '') else 500
        return jsonify(prediction_result), status_code

if __name__ == '__main__':
    # In a Colab environment, use 0.0.0.0 to make it accessible
    # For local development, '127.0.0.1' or 'localhost' is typical.
    # debug=True should be False in production.
    app.run(host='0.0.0.0', port=5000, debug=True)