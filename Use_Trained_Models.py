# ============================================================
# 🤖 How to Use Your Trained Models
# ============================================================
# This script demonstrates how to load and use the trained
# models that were saved from Housing Price.py
# ============================================================

import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

# ============================================================
# 📚 Load Example Data
# ============================================================

# Option 1: Use your dataset
data = pd.read_csv('house_price_regression_dataset.csv')
X = data.drop('House_Price', axis=1)

# Option 2: Use a single sample
sample_data = X.iloc[0:1]  # Get first row as DataFrame
print("📊 Sample Input Data:")
print(sample_data)
print()

# ============================================================
# 🤖 Option A: Using Scikit-Learn Model
# ============================================================

print("=" * 60)
print("🟢 OPTION A: Scikit-Learn Linear Regression Model")
print("=" * 60 + "\n")

try:
    # Load the saved model and scaler
    sklearn_model = joblib.load('sklearn_linear_model.pkl')
    print("✅ Scikit-Learn model loaded successfully!\n")
    
    # Make predictions
    predictions = sklearn_model.predict(X.iloc[0:5])
    
    print("🔮 Predictions for first 5 samples:")
    for i, pred in enumerate(predictions):
        print(f"   Sample {i+1}: ${pred:,.2f}")
    print()
    
except FileNotFoundError:
    print("❌ Model files not found. Please run 'Housing Price.py' first!\n")

# ============================================================
# 🤖 Option B: Using TensorFlow Model
# ============================================================

print("=" * 60)
print("🔵 OPTION B: TensorFlow Neural Network Model")
print("=" * 60 + "\n")

try:
    # Load the saved scaler and TensorFlow model
    scaler = joblib.load('scaler.pkl')
    tf_model = keras.models.load_model('tensorflow_neural_network.h5')
    print("✅ TensorFlow model loaded successfully!\n")
    
    # Scale the input data (IMPORTANT!)
    X_scaled = scaler.transform(X.iloc[0:5])
    
    # Make predictions
    predictions = tf_model.predict(X_scaled, verbose=0)
    predictions = predictions.flatten()
    
    print("🔮 Predictions for first 5 samples:")
    for i, pred in enumerate(predictions):
        print(f"   Sample {i+1}: ${pred:,.2f}")
    print()
    
except FileNotFoundError:
    print("❌ Model files not found. Please run 'Housing Price.py' first!\n")

# ============================================================
# 📋 Prediction on New Data
# ============================================================

print("=" * 60)
print("🆕 Making Predictions on New Data")
print("=" * 60 + "\n")

# Example: Create a new house sample
# Adjust these values based on your dataset features
new_house = pd.DataFrame({
    X.columns[0]: [1500],  # Replace with actual feature value
    X.columns[1]: [3],     # Replace with actual feature value
    # ... add all your features
})

print("📝 New House Data:")
print(new_house)
print()

try:
    # Scikit-Learn prediction
    sklearn_pred = sklearn_model.predict(new_house)[0]
    print(f"🟢 Scikit-Learn Prediction: ${sklearn_pred:,.2f}")
    
    # TensorFlow prediction
    new_house_scaled = scaler.transform(new_house)
    tf_pred = tf_model.predict(new_house_scaled, verbose=0)[0][0]
    print(f"🔵 TensorFlow Prediction: ${tf_pred:,.2f}")
    
except Exception as e:
    print(f"⚠️  Error: {e}")

print()

# ============================================================
# 💡 Tips & Best Practices
# ============================================================

print("=" * 60)
print("💡 Best Practices")
print("=" * 60)
print("""
1️⃣ ALWAYS SCALE INPUT DATA FOR TENSORFLOW:
   - Use the scaler that was fitted during training
   - Must match the original scaling transformation

2️⃣ DATA FORMAT:
   - Use pandas DataFrames for consistency
   - Ensure feature columns match training data

3️⃣ BATCH PREDICTIONS:
   - Feed multiple samples at once for efficiency
   - Set verbose=0 in TensorFlow to suppress output

4️⃣ ERROR HANDLING:
   - Always check if model files exist
   - Validate input data shape and types

5️⃣ MODEL SELECTION:
   - Scikit-Learn: Faster, simpler, good for linear relationships
   - TensorFlow: Better for complex patterns, but slower

6️⃣ DEPLOYMENT:
   - Save models after training
   - Version your models for reproducibility
   - Keep scaler with the TensorFlow model

7️⃣ PREDICTIONS IN PRODUCTION:
   sklearn_pred = model.predict(X)  # Returns array
   
   tf_pred = model.predict(X, verbose=0)  # Returns array
   tf_pred = tf_pred.flatten()  # Convert to 1D if needed

8️⃣ GET PREDICTION CONFIDENCE:
   # For neural networks, you can get intermediate layer outputs
   # or use ensemble methods for uncertainty estimation
""")
