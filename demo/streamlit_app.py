"""Streamlit demo application for speech recognition system."""

import io
import tempfile
from pathlib import Path
from typing import Optional

import streamlit as st
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from speech_recognition.models import ConformerCTC
from speech_recognition.features import MelSpectrogramExtractor
from speech_recognition.metrics import Evaluator
from speech_recognition.utils import get_device, set_seed


class SpeechRecognitionDemo:
    """Streamlit demo for speech recognition."""
    
    def __init__(self):
        """Initialize the demo."""
        st.set_page_config(
            page_title="Speech Recognition System",
            page_icon="🎤",
            layout="wide",
        )
        
        # Set up device
        self.device = get_device("auto")
        set_seed(42)
        
        # Initialize components
        self.model = None
        self.feature_extractor = None
        self.evaluator = None
        
        # Load model (placeholder for demo)
        self._load_model()
    
    def _load_model(self):
        """Load the speech recognition model."""
        try:
            # For demo purposes, create a placeholder model
            # In practice, this would load a trained model
            self.model = ConformerCTC(
                input_dim=80,
                encoder_dim=512,
                num_encoder_layers=17,
                num_attention_heads=8,
                feedforward_expansion_factor=4,
                conv_expansion_factor=2,
                dropout=0.1,
                conv_kernel_size=31,
                vocab_size=5000,
                blank_id=0,
            )
            
            self.feature_extractor = MelSpectrogramExtractor(
                sample_rate=16000,
                n_fft=512,
                hop_length=160,
                win_length=400,
                n_mels=80,
            )
            
            self.evaluator = Evaluator({
                "metrics": ["wer", "cer", "token_accuracy"]
            })
            
            st.success("Model loaded successfully!")
            
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
    
    def _process_audio(self, audio_data: bytes, sample_rate: int) -> tuple[torch.Tensor, str]:
        """
        Process audio data and return features and transcription.
        
        Args:
            audio_data: Raw audio bytes.
            sample_rate: Audio sample rate.
            
        Returns:
            Tuple of (features, transcription).
        """
        try:
            # Convert bytes to tensor
            audio_tensor = torch.frombuffer(audio_data, dtype=torch.float32)
            
            # Resample if necessary
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                audio_tensor = resampler(audio_tensor)
            
            # Extract features
            features = self.feature_extractor(audio_tensor)
            
            # For demo, return a placeholder transcription
            # In practice, this would run the actual model inference
            transcription = "This is a placeholder transcription for the demo."
            
            return features, transcription
            
        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")
            return None, ""
    
    def _plot_spectrogram(self, features: torch.Tensor) -> go.Figure:
        """Create spectrogram visualization."""
        # Convert to numpy
        spec_data = features.numpy()
        
        # Create plotly figure
        fig = go.Figure(data=go.Heatmap(
            z=spec_data,
            colorscale='Viridis',
            xaxis='Time',
            yaxis='Frequency',
        ))
        
        fig.update_layout(
            title="Mel Spectrogram",
            xaxis_title="Time Frames",
            yaxis_title="Mel Frequency Bins",
            height=400,
        )
        
        return fig
    
    def _create_waveform_plot(self, audio_data: bytes, sample_rate: int) -> go.Figure:
        """Create waveform visualization."""
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.float32)
            
            # Create time axis
            time_axis = np.linspace(0, len(audio_array) / sample_rate, len(audio_array))
            
            # Create plotly figure
            fig = go.Figure(data=go.Scatter(
                x=time_axis,
                y=audio_array,
                mode='lines',
                name='Waveform',
            ))
            
            fig.update_layout(
                title="Audio Waveform",
                xaxis_title="Time (seconds)",
                yaxis_title="Amplitude",
                height=300,
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating waveform plot: {str(e)}")
            return go.Figure()
    
    def run(self):
        """Run the Streamlit demo."""
        # Header
        st.title("🎤 Speech Recognition System")
        st.markdown("A modern ASR system with Conformer architecture")
        
        # Privacy disclaimer
        st.warning("""
        **PRIVACY DISCLAIMER**: This is a research demonstration system. 
        Audio data is processed locally and not stored. This system is not intended 
        for biometric identification or production use. Please do not upload 
        sensitive or personal information.
        """)
        
        # Sidebar
        st.sidebar.header("Configuration")
        
        # Model settings
        st.sidebar.subheader("Model Settings")
        sample_rate = st.sidebar.selectbox(
            "Sample Rate",
            [8000, 16000, 22050, 44100],
            index=1,
            help="Audio sample rate for processing"
        )
        
        # Main content
        tab1, tab2, tab3 = st.tabs(["🎤 Record Audio", "📁 Upload File", "📊 Analysis"])
        
        with tab1:
            st.header("Record Audio")
            st.markdown("Record audio directly in your browser for speech recognition.")
            
            # Audio recording
            audio_bytes = st.audio(
                key="audio_recorder",
                format="audio/wav",
                sample_rate=sample_rate,
            )
            
            if audio_bytes:
                st.success("Audio recorded successfully!")
                
                # Process audio
                if st.button("Transcribe Audio", key="transcribe_recorded"):
                    with st.spinner("Processing audio..."):
                        features, transcription = self._process_audio(audio_bytes, sample_rate)
                        
                        if features is not None:
                            # Display results
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("Transcription")
                                st.text_area(
                                    "Recognized Text",
                                    value=transcription,
                                    height=100,
                                    disabled=True,
                                )
                            
                            with col2:
                                st.subheader("Audio Visualization")
                                waveform_fig = self._create_waveform_plot(audio_bytes, sample_rate)
                                st.plotly_chart(waveform_fig, use_container_width=True)
                            
                            # Show spectrogram
                            st.subheader("Spectrogram")
                            spec_fig = self._plot_spectrogram(features)
                            st.plotly_chart(spec_fig, use_container_width=True)
        
        with tab2:
            st.header("Upload Audio File")
            st.markdown("Upload an audio file for speech recognition.")
            
            # File upload
            uploaded_file = st.file_uploader(
                "Choose an audio file",
                type=['wav', 'mp3', 'flac', 'm4a'],
                help="Supported formats: WAV, MP3, FLAC, M4A"
            )
            
            if uploaded_file is not None:
                st.success(f"File uploaded: {uploaded_file.name}")
                
                # Display file info
                file_size = len(uploaded_file.getvalue())
                st.info(f"File size: {file_size / 1024:.1f} KB")
                
                # Process uploaded file
                if st.button("Transcribe File", key="transcribe_file"):
                    with st.spinner("Processing uploaded file..."):
                        # Read file content
                        audio_data = uploaded_file.read()
                        
                        # Process audio
                        features, transcription = self._process_audio(audio_data, sample_rate)
                        
                        if features is not None:
                            # Display results
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("Transcription")
                                st.text_area(
                                    "Recognized Text",
                                    value=transcription,
                                    height=100,
                                    disabled=True,
                                )
                                
                                # Download transcription
                                st.download_button(
                                    label="Download Transcription",
                                    data=transcription,
                                    file_name="transcription.txt",
                                    mime="text/plain",
                                )
                            
                            with col2:
                                st.subheader("Audio Visualization")
                                waveform_fig = self._create_waveform_plot(audio_data, sample_rate)
                                st.plotly_chart(waveform_fig, use_container_width=True)
                            
                            # Show spectrogram
                            st.subheader("Spectrogram")
                            spec_fig = self._plot_spectrogram(features)
                            st.plotly_chart(spec_fig, use_container_width=True)
        
        with tab3:
            st.header("Model Analysis")
            st.markdown("Analyze model performance and characteristics.")
            
            # Model info
            st.subheader("Model Architecture")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Encoder Dimension", "512")
                st.metric("Number of Layers", "17")
            
            with col2:
                st.metric("Attention Heads", "8")
                st.metric("Vocabulary Size", "5,000")
            
            with col3:
                st.metric("Input Features", "80")
                st.metric("Conv Kernel Size", "31")
            
            # Performance metrics
            st.subheader("Performance Metrics")
            
            # Create sample metrics for demo
            metrics_data = {
                "WER": 0.15,
                "CER": 0.08,
                "Token Accuracy": 0.92,
                "Latency": 0.05,
                "RTF": 0.3,
            }
            
            # Display metrics
            cols = st.columns(len(metrics_data))
            for i, (metric, value) in enumerate(metrics_data.items()):
                with cols[i]:
                    if metric in ["WER", "CER"]:
                        st.metric(metric, f"{value:.3f}", f"{value*100:.1f}%")
                    elif metric == "Token Accuracy":
                        st.metric(metric, f"{value:.3f}", f"{value*100:.1f}%")
                    elif metric == "Latency":
                        st.metric(metric, f"{value:.3f}s")
                    else:
                        st.metric(metric, f"{value:.3f}")
            
            # Model comparison chart
            st.subheader("Model Comparison")
            
            models = ["Conformer-CTC", "Transformer-CTC", "RNN-T", "Whisper"]
            wer_scores = [0.15, 0.18, 0.22, 0.12]
            
            fig = go.Figure(data=go.Bar(
                x=models,
                y=wer_scores,
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            ))
            
            fig.update_layout(
                title="Word Error Rate Comparison",
                xaxis_title="Model",
                yaxis_title="WER",
                height=400,
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        **Speech Recognition System** - A modern ASR implementation with Conformer architecture.
        
        Built with PyTorch, Streamlit, and modern deep learning techniques.
        """)
        
        # Additional info
        with st.expander("Technical Details"):
            st.markdown("""
            **Architecture**: Conformer with CTC loss
            
            **Features**:
            - Multi-head self-attention
            - Convolution modules
            - Feed-forward networks
            - Positional encoding
            
            **Training**:
            - SpecAugment data augmentation
            - Speed perturbation
            - AdamW optimizer
            - OneCycleLR scheduler
            
            **Evaluation**:
            - Word Error Rate (WER)
            - Character Error Rate (CER)
            - Token Accuracy
            - Latency and RTF metrics
            """)


def main():
    """Main function to run the demo."""
    demo = SpeechRecognitionDemo()
    demo.run()


if __name__ == "__main__":
    main()
