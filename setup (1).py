#!/usr/bin/env python3
"""
Setup script for House Price Prediction Project
Run this once to set up everything
"""

import os
import sys
import subprocess
import webbrowser

def print_header():
    print("\n" + "="*60)
    print("🏠 HOUSE PRICE PREDICTION PROJECT SETUP")
    print("="*60)

def check_python():
    print("\n🔍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print(f"❌ Python 3.7+ required. You have {version.major}.{version.minor}")
        return False
    print(f" Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def install_requirements():
    print("\n Installing requirements...")
    requirements = [
        "flask==2.3.3",
        "flask-cors==4.0.0",
        "scikit-learn==1.3.0",
        "pandas==2.0.3",
        "numpy==1.24.3",
        "joblib==1.3.2",
        "matplotlib==3.7.2"
    ]
    
    for package in requirements:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("✅ All requirements installed")

def train_model():
    print("\n🤖 Training the machine learning model...")
    
    # Check if training script exists
    if os.path.exists("hacketon_houseprediction_simple.py"):
        print("Running hacketon_houseprediction_simple.py...")
        subprocess.check_call([sys.executable, "hacketon_houseprediction_simple.py"])
    elif os.path.exists("hacketon_houseprediction.py"):
        print("Running hacketon_houseprediction.py...")
        subprocess.check_call([sys.executable, "hacketon_houseprediction.py"])
    else:
        print("⚠️ No training script found. Creating a simple model...")
        # Create a simple model directly
        from sklearn.linear_model import LinearRegression
        import joblib
        import numpy as np
        
        model = LinearRegression()
        X = np.array([
            [1000, 2, 1.0, 3],
            [1500, 3, 2.0, 3],
            [2000, 3, 2.0, 4],
            [2500, 4, 2.5, 4],
            [3000, 4, 3.0, 5]
        ])
        y = np.array([300000, 450000, 600000, 750000, 900000])
        model.fit(X, y)
        
        joblib.dump(model, 'house_price_model_simple.pkl')
        print("✅ Simple model created and saved")
    
    # Check if model was created
    if os.path.exists("house_price_model_simple.pkl") or os.path.exists("house_price_prediction_model.pkl"):
        print("✅ Model training complete")
    else:
        print("⚠️ Model file not created. Backend will create one on startup.")

def start_backend():
    print("\n🚀 Starting backend server...")
    print("\n📋 IMPORTANT: Keep this terminal open!")
    print("   The backend server will run here.")
    print("\n🌐 Backend URL: http://127.0.0.1:5000")
    print("📁 Open 'index.html' in your browser or run:")
    print("   python -m http.server 8000")
    print("\n🎯 Then navigate to: http://localhost:8000")
    print("="*60)
    
    # Start backend
    subprocess.check_call([sys.executable, "backend.py"])

def main():
    print_header()
    
    if not check_python():
        return
    
    try:
        # Step 1: Install requirements
        install_requirements()
        
        # Step 2: Train model
        train_model()
        
        # Step 3: Start backend
        start_backend()
        
    except KeyboardInterrupt:
        print("\n\n👋 Setup interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during setup: {e}")
        print("\n💡 Try running these commands manually:")
        print("1. pip install -r requirements.txt")
        print("2. python hacketon_houseprediction_simple.py")
        print("3. python backend.py")
        print("4. Open index.html in browser")

if __name__ == "__main__":
    main()