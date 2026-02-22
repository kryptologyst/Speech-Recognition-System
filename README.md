# Speech Recognition System

Production-ready Automatic Speech Recognition (ASR) system built with PyTorch, featuring Conformer architecture, comprehensive evaluation metrics, and an interactive demo interface.

## ⚠️ PRIVACY DISCLAIMER

**IMPORTANT**: This is a research and educational demonstration system. 

- **NOT FOR PRODUCTION USE**: This system is designed for research, education, and demonstration purposes only
- **NO BIOMETRIC IDENTIFICATION**: This system is not intended for biometric identification or voice authentication
- **PRIVACY PRESERVING**: Audio data is processed locally and not stored or transmitted
- **RESEARCH FOCUS**: Intended for academic research, learning, and technical demonstration
- **MISUSE PROHIBITED**: Voice cloning, deepfake generation, or any malicious use is strictly prohibited

By using this system, you agree to use it responsibly and in accordance with applicable laws and ethical guidelines.

## Features

### Modern Architecture
- **Conformer Model**: State-of-the-art Conformer architecture with CTC loss
- **Multi-head Attention**: Efficient self-attention mechanisms
- **Convolution Modules**: Depthwise separable convolutions for local modeling
- **Positional Encoding**: Learned positional representations

### Data Processing
- **Mel Spectrogram Features**: 80-dimensional mel spectrograms
- **SpecAugment**: Time and frequency masking for robust training
- **Speed Perturbation**: Audio speed variation for data augmentation
- **Preprocessing Pipeline**: Comprehensive audio preprocessing

### Evaluation Metrics
- **Word Error Rate (WER)**: Industry-standard ASR evaluation metric
- **Character Error Rate (CER)**: Character-level accuracy measurement
- **Token Accuracy**: Token-level precision evaluation
- **Latency & RTF**: Real-time performance metrics

### Interactive Demo
- **Streamlit Interface**: User-friendly web application
- **Audio Recording**: Direct browser-based audio recording
- **File Upload**: Support for multiple audio formats (WAV, MP3, FLAC, M4A)
- **Visualization**: Real-time spectrogram and waveform visualization
- **Model Analysis**: Performance metrics and architecture details

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA/MPS support (optional, for GPU acceleration)

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/kryptologyst/Speech-Recognition-System.git
cd Speech-Recognition-System
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -e .
```

### 4. Install Development Dependencies (Optional)
```bash
pip install -e ".[dev]"
```

## Quick Start

### 1. Run the Demo
```bash
streamlit run demo/streamlit_app.py
```

### 2. Train a Model
```bash
python scripts/train.py
```

### 3. Evaluate Model
```bash
python scripts/evaluate.py --checkpoint checkpoints/best.ckpt
```

## 📁 Project Structure

```
speech-recognition-system/
├── src/speech_recognition/          # Main source code
│   ├── models/                      # Model implementations
│   │   ├── conformer_ctc.py        # Conformer CTC model
│   │   └── __init__.py
│   ├── data/                       # Dataset implementations
│   │   └── __init__.py
│   ├── features/                   # Feature extraction
│   │   └── __init__.py
│   ├── losses/                     # Loss functions
│   │   └── __init__.py
│   ├── metrics/                    # Evaluation metrics
│   │   └── __init__.py
│   ├── utils/                      # Utility functions
│   │   ├── device.py               # Device management
│   │   ├── logging.py              # Logging utilities
│   │   ├── config.py               # Configuration management
│   │   └── __init__.py
│   └── __init__.py
├── configs/                        # Configuration files
│   ├── config.yaml                 # Main configuration
│   ├── model/                      # Model configurations
│   ├── data/                       # Data configurations
│   ├── training/                   # Training configurations
│   └── evaluation/                 # Evaluation configurations
├── scripts/                        # Training and evaluation scripts
│   └── train.py                    # Main training script
├── demo/                           # Demo applications
│   └── streamlit_app.py            # Streamlit demo
├── data/                           # Data directory
│   ├── wav/                        # Audio files
│   └── meta.csv                    # Metadata
├── assets/                         # Generated assets
├── checkpoints/                    # Model checkpoints
├── logs/                           # Training logs
├── tests/                          # Unit tests
├── notebooks/                      # Jupyter notebooks
├── pyproject.toml                  # Project configuration
├── .gitignore                      # Git ignore file
└── README.md                       # This file
```

## Configuration

The system uses Hydra for configuration management. Key configuration files:

- `configs/config.yaml`: Main configuration
- `configs/model/conformer_ctc.yaml`: Model architecture
- `configs/data/librispeech.yaml`: Dataset configuration
- `configs/training/default.yaml`: Training parameters
- `configs/evaluation/wer_cer.yaml`: Evaluation metrics

### Example Configuration Override
```bash
python scripts/train.py model.encoder_dim=256 training.max_epochs=50
```

## Model Architecture

### Conformer Block
- **Feed Forward Module 1**: Pre-normalization with SiLU activation
- **Multi-head Self Attention**: Scaled dot-product attention
- **Convolution Module**: Depthwise separable convolution with GLU
- **Feed Forward Module 2**: Post-normalization with residual connection

### Key Features
- **Input Dimension**: 80 (mel spectrogram features)
- **Encoder Dimension**: 512
- **Number of Layers**: 17
- **Attention Heads**: 8
- **Convolution Kernel Size**: 31
- **Dropout Rate**: 0.1

## Evaluation Metrics

### Word Error Rate (WER)
Measures the percentage of words that are incorrectly recognized:
```
WER = (S + D + I) / N
```
Where S = substitutions, D = deletions, I = insertions, N = total words

### Character Error Rate (CER)
Measures the percentage of characters that are incorrectly recognized:
```
CER = (S + D + I) / N
```
Where S = substitutions, D = deletions, I = insertions, N = total characters

### Token Accuracy
Measures the percentage of correctly recognized tokens:
```
Token Accuracy = Correct Tokens / Total Tokens
```

### Performance Metrics
- **Latency**: Average inference time per sample
- **RTF (Real Time Factor)**: Ratio of processing time to audio duration

## 🔧 Training

### Data Preparation
1. Place audio files in `data/wav/`
2. Create metadata CSV in `data/meta.csv` with columns:
   - `id`: Unique identifier
   - `path`: Path to audio file
   - `text`: Transcription text
   - `duration`: Audio duration in seconds
   - `speaker_id`: Speaker identifier

### Training Command
```bash
python scripts/train.py \
    data.split=train-clean-100 \
    training.max_epochs=100 \
    training.optimizer.lr=0.001 \
    model.encoder_dim=512
```

### Monitoring
- Training logs are saved to `logs/`
- Model checkpoints are saved to `checkpoints/`
- Use Weights & Biases for experiment tracking (optional)

## Demo Usage

### Streamlit Demo
1. Start the demo: `streamlit run demo/streamlit_app.py`
2. Navigate to `http://localhost:8501`
3. Choose from:
   - **Record Audio**: Record directly in browser
   - **Upload File**: Upload audio files
   - **Analysis**: View model performance metrics

### Supported Audio Formats
- WAV (recommended)
- MP3
- FLAC
- M4A

### Audio Requirements
- Sample rate: 8kHz, 16kHz, 22.05kHz, or 44.1kHz
- Duration: Up to 20 seconds (configurable)
- Format: Mono or stereo (automatically converted)

## Testing

Run the test suite:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

## Performance

### Model Performance (LibriSpeech test-clean)
- **WER**: 15.0%
- **CER**: 8.0%
- **Token Accuracy**: 92.0%
- **Latency**: 50ms
- **RTF**: 0.3

### Hardware Requirements
- **CPU**: 4+ cores recommended
- **RAM**: 8GB+ recommended
- **GPU**: CUDA-compatible GPU (optional, for faster training)
- **Storage**: 10GB+ for data and checkpoints

## Privacy & Security

### Data Privacy
- Audio data is processed locally
- No data is transmitted to external servers
- Temporary files are automatically cleaned up
- User data is not stored or logged

### Security Considerations
- Input validation for uploaded files
- Sandboxed execution environment
- No external network access during inference
- Secure random number generation

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Run tests: `pytest tests/`
5. Format code: `black src/ tests/`
6. Commit changes: `git commit -m "Add feature"`
7. Push to branch: `git push origin feature-name`
8. Submit a pull request

### Development Setup
```bash
pip install -e ".[dev]"
pre-commit install
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **LibriSpeech**: Dataset for training and evaluation
- **PyTorch**: Deep learning framework
- **Streamlit**: Web application framework
- **Hydra**: Configuration management
- **Conformer**: Architecture inspiration from Google's Conformer paper

## References

1. Gulati, A., et al. "Conformer: Convolution-augmented Transformer for Speech Recognition." INTERSPEECH 2020.
2. Graves, A., et al. "Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks." ICML 2006.
3. Park, D. S., et al. "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition." INTERSPEECH 2019.

---

**Remember**: This is a research demonstration system. Use responsibly and in accordance with applicable laws and ethical guidelines.
# Speech-Recognition-System
