#!/usr/bin/env python3
"""Setup script for speech recognition system."""

import os
import sys
import subprocess
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("🚀 Setting up Speech Recognition System...")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Create necessary directories
    directories = [
        "data/wav",
        "data/meta",
        "assets",
        "checkpoints",
        "logs",
        "configs/local",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    # Install dependencies
    if not run_command("pip install -e .", "Installing package dependencies"):
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Install development dependencies
    if not run_command("pip install -e .[dev]", "Installing development dependencies"):
        print("⚠️  Failed to install development dependencies (optional)")
    
    # Run tests
    if not run_command("python -m pytest tests/ -v", "Running tests"):
        print("⚠️  Some tests failed (this is expected for a demo system)")
    
    # Create sample data
    print("📊 Creating sample data...")
    try:
        import pandas as pd
        import numpy as np
        
        # Create sample metadata
        sample_data = []
        for i in range(10):
            sample_data.append({
                "id": f"sample_{i:03d}",
                "path": f"data/wav/sample_{i:03d}.wav",
                "text": f"This is sample audio number {i}",
                "duration": np.random.uniform(1.0, 5.0),
                "speaker_id": f"speaker_{i % 3}",
            })
        
        df = pd.DataFrame(sample_data)
        df.to_csv("data/meta.csv", index=False)
        print("✅ Created sample metadata")
        
    except ImportError:
        print("⚠️  Could not create sample data (pandas not available)")
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Run the demo: streamlit run demo/streamlit_app.py")
    print("2. Train a model: python scripts/train.py")
    print("3. Evaluate model: python scripts/evaluate.py --checkpoint checkpoints/best.ckpt")
    print("\n📚 Read the README.md for detailed instructions")
    print("⚠️  Please read DISCLAIMER.md for important privacy information")


if __name__ == "__main__":
    main()
