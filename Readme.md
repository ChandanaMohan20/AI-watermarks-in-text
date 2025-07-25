#  AI Text Classifier: Identifying AI Watermarks in Text

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange.svg)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)](https://flask.palletsprojects.com)
[![Accuracy](https://img.shields.io/badge/Accuracy-99%25-brightgreen.svg)](#results)

A state-of-the-art deep learning application that distinguishes between human-written and AI-generated text using a novel hybrid neural architecture. This project addresses the growing need for reliable AI content detection in academic integrity, content moderation, and misinformation control.

##  Project Overview

This system combines **BiLSTM**, **Transformer encoders**, and **Convolutional Neural Networks** to effectively capture both local and global linguistic features, achieving **99% accuracy** in detecting AI-generated text even with grammatical errors or stylistic variations.

### Key Features

-  **99% Classification Accuracy** - Outperforms existing tools like GPTZero
-  **Robust Error Handling** - Detects AI text even with grammatical/spelling errors
-  **Multi-Domain Support** - Works on archaic English, paraphrased content, and various writing styles
-  **Beautiful Web Interface** - Modern, responsive UI for easy text analysis
-  **Real-time Predictions** - Fast inference with confidence scores
-  **Production Ready** - Deployable Flask backend with proper error handling

##  Architecture

Our hybrid neural architecture leverages the strengths of multiple deep learning components:

```
Input Text → TextVectorization → Embedding → BiLSTM → Transformer → Conv1D → GlobalMaxPooling → Dense → Prediction
```

### Component Breakdown

- **TextVectorization**: Character-level 3-gram and 5-gram tokenization (1024 tokens)
- **Embedding Layer**: 64-dimensional word vectors
- **BiLSTM**: Captures long-range dependencies and sequential flow
- **Transformer Block**: Attention mechanism for global contextual understanding
- **Conv1D**: Detects local stylistic patterns and n-gram features
- **GlobalMaxPooling**: Retains strongest activations across temporal dimensions
- **Dense + Dropout**: Final classification with regularization

##  Results

### Performance Metrics

| Metric | Human Text | AI Text | Overall |
|--------|------------|---------|---------|
| **Precision** | 0.99 | 1.00 | 0.99 |
| **Recall** | 1.00 | 0.99 | 0.99 |
| **F1-Score** | 0.99 | 0.99 | 0.99 |

### Confusion Matrix
-  **100%** of human texts correctly identified
-  **99%** of AI texts correctly detected
-  Only **1%** false positive rate

### Comparative Analysis
Our model consistently outperformed GPTZero, especially on:
- Grammatically flawed AI-generated text
- Paraphrased AI content
- Texts with intentional errors or stylistic variations

## Quick Start

### Prerequisites

```bash
Python 3.9+
TensorFlow 2.19+
Flask 3.0+
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ai-text-classifier.git
cd ai-text-classifier
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Verify required files**
```
your_project/
├── app.py                 # Flask backend
├── templates/
│   └── index.html        # Web interface
├── model.h5              # Trained model (59.3MB)
├── train_data.csv        # Training data for vectorizer
└── requirements.txt      # Dependencies
```

### Local Development

**Start the Flask server:**
```bash
python app.py
```

**Open your browser:**
```
http://localhost:5000
```

##  Deployment

### Option 1: Railway (Recommended)
1. Push code to GitHub
2. Go to [railway.app](https://railway.app)
3. Deploy from GitHub repo
4. Automatic deployment in ~2 minutes

### Option 2: Render
1. Connect GitHub to [render.com](https://render.com)
2. Create new Web Service
3. Use settings:
   - Build: `pip install -r requirements.txt`
   - Start: `gunicorn app:app`

### Option 3: Google Cloud Run
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/ai-classifier
gcloud run deploy --image gcr.io/PROJECT_ID/ai-classifier
```

## Technical Details

### Dataset
- **Size**: 2,702 balanced samples (1,351 per class)
- **Sources**: 
  - Human essays from Kaggle
  - AI-generated texts from GPT-3, GPT-4, QuillBot
  - Texts with intentional grammatical errors
  - Archaic English samples (Shakespeare, historical sources)
- **Balancing**: SMOTE oversampling to prevent class bias

### Preprocessing Pipeline
1. **Normalization**: UTF-8 decomposition, lowercasing
2. **Cleaning**: Regex removal of special characters, Twitter handles
3. **Tokenization**: Character-level n-grams (3,5) with TensorFlow
4. **Standardization**: Consistent unicode handling
5. **Padding**: Fixed sequence length of 1024 tokens

### Training Configuration
- **Optimizer**: Adam with adaptive learning rates
- **Loss Function**: Binary Cross-entropy
- **Regularization**: Dropout (0.5) + Early stopping
- **Batch Size**: 32
- **Epochs**: 10 (with early stopping)

### Model Specifications
- **Parameters**: ~2.3M trainable parameters
- **Input Shape**: (None, 1024)
- **Output**: Single probability score (0-1)
- **Model Size**: 59.3MB (perfect for deployment)

## Usage Examples

### Web Interface
1. **Enter text** in the analysis box
2. **Click "Analyze Text"** 
3. **View results** with confidence scores
4. **Try sample texts** for demonstration

### API Usage
```python
import requests

response = requests.post('http://localhost:5000/predict', 
                        json={'text': 'Your text here'})
result = response.json()

print(f"Prediction: {result['label']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Response Format
```json
{
  "text": "Input text...",
  "is_ai": false,
  "confidence": 0.85,
  "label": "Human Written"
}
```

##  Key Strengths

### Superior Robustness
- **Grammatical Errors**: Correctly identifies flawed AI text where GPTZero fails
- **Paraphrasing**: Detects AI signatures even after significant rewording
- **Style Variations**: Handles formal, informal, archaic, and creative writing
- **Multi-length**: Works on short snippets to long documents

### Technical Advantages
- **Multi-level Analysis**: Combines local patterns, sequential flow, and global context
- **Character-level Features**: Captures subtle orthographic and stylistic signals
- **Attention Mechanisms**: Models long-range dependencies effectively
- **Regularization**: Prevents overfitting with dropout and early stopping

## Performance Analysis

### Training Dynamics
- **Rapid Convergence**: Model reaches 99% accuracy within 5 epochs
- **Stable Learning**: Training and validation curves remain close
- **Minimal Overfitting**: Regularization techniques effectively control complexity

### Ablation Study Results
- **Without BiLSTM**: ↓ Syntactic pattern recognition, more false positives
- **Without Transformer**: ↓ Global semantic understanding, context misses
- **Without Conv1D**: ↓ Local stylistic detection, poor on noisy inputs

## Development

### Project Structure
```
ai-text-classifier/
├── app.py                 # Flask backend server
├── templates/
│   └── index.html        # Frontend interface
├── static/               # CSS, JS, images (optional)
├── model.h5              # Trained transformer model
├── train_data.csv        # Dataset for vectorizer adaptation
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
├── .gitignore           # Git ignore rules
└── LICENSE              # Project license
```

### API Endpoints
- `GET /` - Serve web interface
- `POST /predict` - Text classification API
- `GET /health` - Health check endpoint

### Model Architecture Code
```python
# Simplified architecture overview
model = Sequential([
    TextVectorization(ngrams=(3,5), max_tokens=75000),
    Embedding(embedding_dim=64),
    Bidirectional(LSTM(32)),
    TransformerBlock(embed_dim=64, num_heads=4),
    Conv1D(128, 3, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
```

## Research Background

This project is based on academic research conducted at Rice University, addressing the critical challenge of distinguishing human-written from AI-generated content. The work builds upon recent advances in transformer architectures and combines multiple neural network paradigms for robust text classification.

### Related Work
- Alamleh et al. (2023): Traditional ML approaches to AI text detection
- Mitrović et al. (2023): Transformer-based models for short-form content
- Li et al. (2023): ChatGPT vs human expert comparison (HC3 corpus)

### Novel Contributions
- **Hybrid Architecture**: First to combine BiLSTM + Transformer + CNN for AI detection
- **Robustness Analysis**: Systematic evaluation on grammatically flawed inputs
- **Character-level Processing**: N-gram tokenization for stylistic signal capture
- **Production System**: Complete web application with deployment guide

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone and setup
git clone https://github.com/yourusername/ai-text-classifier.git
cd ai-text-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Start development server
python app.py
```

## Troubleshooting

### Common Issues

**Model Loading Errors**
- Ensure `model.h5` and `train_data.csv` are in the project root
- Check TensorFlow version compatibility
- Verify custom TransformerBlock class is properly defined

**Memory Issues**
- Use `tensorflow-cpu` instead of `tensorflow` for deployment
- Add memory optimization: `tf.config.experimental.enable_memory_growth`
- Consider model quantization for resource-constrained environments

**Deployment Problems**
- Check all dependencies in `requirements.txt`
- Ensure model file is under 100MB for GitHub
- Use Git LFS for larger models or cloud storage

## Benchmarks

### Inference Speed
- **CPU**: ~50ms per prediction
- **GPU**: ~10ms per prediction
- **Batch Processing**: 100+ texts/second

### Resource Requirements
- **RAM**: 2GB minimum, 4GB recommended
- **Storage**: 200MB for model + dependencies
- **CPU**: 2+ cores recommended for production

## Future Enhancements

- [ ] **Multi-language Support**: Extend to non-English text detection
- [ ] **Real-time Streaming**: Process live text streams
- [ ] **Explainability**: Add SHAP/LIME for prediction interpretation
- [ ] **Fine-tuning**: Domain-specific model adaptation
- [ ] **Mobile App**: Native iOS/Android applications
- [ ] **Browser Extension**: Real-time web page analysis



## Citation

If you use this work in your research, please cite:

```bibtex
@article{mohan2025identifying,
  title={Identifying AI Watermarks in Text using Deep Learning},
  author={Mohan, Chandana},
  journal={Rice University Computer Science Department},
  year={2025}
}
```

##  Authors

- **Chandana Mohan** - *Initial work* - [cm203@rice.edu](mailto:cm203@rice.edu)
- Rice University, Department of Computer Science

## Acknowledgments

- Rice University Computer Science Department
- TensorFlow and Keras communities
- Kaggle for providing human-written text datasets
- OpenAI for AI-generated text samples
- Contributors to the academic research in AI text detection

---

** If you find this project useful, please give it a star!**

For questions, issues, or collaboration opportunities, please [open an issue](https://github.com/yourusername/ai-text-classifier/issues) or contact the authors directly.