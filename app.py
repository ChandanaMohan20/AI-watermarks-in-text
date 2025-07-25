from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import keras_core as keras
import tensorflow_text as tf_text
import warnings
warnings.filterwarnings("ignore")

# Set Keras backend
os.environ['KERAS_BACKEND'] = 'tensorflow'

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Global variables for model and vectorizer
model = None
vectorize_layer = None

def Clean(text):
    """Text cleaning function from your original code"""
    text = tf_text.normalize_utf8(text, 'NFKD')
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')
    text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
    text = tf.strings.strip(text)
    text = tf.strings.regex_replace(text, '\.\.\.', ' ')
    text = tf.strings.join(['',text, ''], separator=' ')
    return text

def load_model_and_vectorizer():
    """Load the trained model and prepare vectorizer"""
    global model, vectorize_layer
    
    try:
        # Load training data to adapt vectorizer
        train_data = pd.read_csv('train_data.csv')
        
        # Set up vectorization layer (same as your training)
        max_features = 75000
        sequence_length = 512 * 2
        vectorize_layer = tf.keras.layers.TextVectorization(
            standardize=Clean,
            max_tokens=max_features,
            ngrams=(3, 5),
            output_mode="int",
            output_sequence_length=sequence_length,
            pad_to_max_tokens=True
        )
        vectorize_layer.adapt(train_data['text'])
        
        # Load the trained model
        from tensorflow.keras.layers import MultiHeadAttention
        
        # Define TransformerBlock class
        class TransformerBlock(tf.keras.layers.Layer):
            def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
                super(TransformerBlock, self).__init__(**kwargs)
                self.embed_dim = embed_dim
                self.num_heads = num_heads
                self.ff_dim = ff_dim
                self.rate = rate

                self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
                self.ffn = tf.keras.Sequential([
                    tf.keras.layers.Dense(ff_dim, activation="relu"),
                    tf.keras.layers.Dense(embed_dim),
                ])
                self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
                self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
                self.dropout1 = tf.keras.layers.Dropout(rate)
                self.dropout2 = tf.keras.layers.Dropout(rate)

            def call(self, inputs, training=None):
                attn_output = self.att(inputs, inputs)
                attn_output = self.dropout1(attn_output, training=training)
                out1 = self.layernorm1(inputs + attn_output)
                ffn_output = self.ffn(out1)
                ffn_output = self.dropout2(ffn_output, training=training)
                return self.layernorm2(out1 + ffn_output)

            def get_config(self):
                config = super(TransformerBlock, self).get_config()
                config.update({
                    "embed_dim": self.embed_dim,
                    "num_heads": self.num_heads,
                    "ff_dim": self.ff_dim,
                    "rate": self.rate,
                })
                return config
        
        # Load model with custom objects
        with tf.keras.utils.custom_object_scope({'TransformerBlock': TransformerBlock}):
            model = tf.keras.models.load_model('model.h5')
        
        print("✅ Model and vectorizer loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False

@app.route('/')
def index():
    """Serve the HTML interface"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for text classification"""
    try:
        # Get text from request
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        if model is None or vectorize_layer is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Preprocess text
        text_preprocessed = vectorize_layer([text])
        
        # Make prediction
        prediction = model.predict(text_preprocessed, verbose=0)
        confidence = float(prediction[0][0])
        
        # Convert to binary prediction
        is_ai = confidence >= 0.5
        
        # Prepare response
        result = {
            'text': text,
            'is_ai': is_ai,
            'confidence': confidence,
            'label': 'AI Generated' if is_ai else 'Human Written'
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'vectorizer_loaded': vectorize_layer is not None
    })

# Load model on startup (for production)
print("🚀 Loading AI Text Classifier...")
load_model_and_vectorizer()

if __name__ == '__main__':
    print("🌐 Starting development server...")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)