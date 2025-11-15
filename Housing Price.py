# ============================================================
# 📦 Import Libraries
# ============================================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import joblib

# ============================================================
# 📂 Load and Inspect Data
# ============================================================

data = pd.read_csv('house_price_regression_dataset.csv')

print("📊 Dataset Preview:")
print(data.head(), "\n")

# ============================================================
# ⚙️ Prepare Features and Target
# ============================================================

X = data.drop('House_Price', axis=1)
y = data['House_Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalize features for TensorFlow model (important!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 🧠 Option 1: Train Scikit-Learn Linear Regression Model
# ============================================================

sklearn_model = LinearRegression()
sklearn_model.fit(X_train, y_train)
y_pred_sklearn = sklearn_model.predict(X_test)

# ============================================================
# 🧠 Option 2: Train TensorFlow/Keras Neural Network Model
# ============================================================

print("� Building TensorFlow Neural Network...\n")

# Create a sequential neural network
tf_model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)  # Output layer for regression
])

# Compile the model
tf_model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# Display model architecture
print("📐 Model Architecture:")
tf_model.summary()
print()

# Train the model
print("🚀 Training TensorFlow Model...\n")
history = tf_model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=16,
    verbose=1
)

# Make predictions
y_pred_tf = tf_model.predict(X_test_scaled, verbose=0)
y_pred_tf = y_pred_tf.flatten()  # Flatten to 1D array

# ============================================================
# 💾 Save Trained Models
# ============================================================

print("\n💾 Saving Models...\n")

# Save scikit-learn model
joblib.dump(sklearn_model, 'sklearn_linear_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("✅ Scikit-Learn model saved as 'sklearn_linear_model.pkl'")
print("✅ Scaler saved as 'scaler.pkl'")

# Save TensorFlow model
tf_model.save('tensorflow_neural_network.h5')
print("✅ TensorFlow model saved as 'tensorflow_neural_network.h5'\n")

# ============================================================
# 📏 Evaluate Model
# ============================================================

mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)
r2_sklearn = r2_score(y_test, y_pred_sklearn)

mse_tf = mean_squared_error(y_test, y_pred_tf)
r2_tf = r2_score(y_test, y_pred_tf)

print("=" * 60)
print("📈 Model Evaluation Comparison")
print("=" * 60)
print("\n🔹 Scikit-Learn Linear Regression:")
print(f"   Mean Squared Error (MSE): {mse_sklearn:.2f}")
print(f"   R-squared (R²): {r2_sklearn:.4f}")

print("\n🔹 TensorFlow Neural Network:")
print(f"   Mean Squared Error (MSE): {mse_tf:.2f}")
print(f"   R-squared (R²): {r2_tf:.4f}")
print("\n" + "=" * 60 + "\n")

# ============================================================
# 🔍 Display Sample Predictions
# ============================================================

comparison = pd.DataFrame({
    'Actual': y_test.values[:10],
    'Predicted (SkLearn)': y_pred_sklearn[:10],
    'Predicted (TensorFlow)': y_pred_tf[:10]
})
print("🔍 Sample Predictions (Actual vs Predicted):")
print(comparison.to_string(index=False))

# ============================================================
# 📊 Graphical Visualizations
# ============================================================

# 1️⃣ Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("🔗 Feature Correlation Heatmap")
plt.show()

# 2️⃣ Actual vs Predicted Scatter Plot (TensorFlow Model)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred_tf, color='dodgerblue', alpha=0.7, edgecolor='k')
plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("🏠 Actual vs Predicted House Prices (TensorFlow)")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()

# 2B️⃣ Comparison: Sklearn vs TensorFlow
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(y_test, y_pred_sklearn, color='green', alpha=0.7, edgecolor='k')
axes[0].set_xlabel("Actual House Price")
axes[0].set_ylabel("Predicted House Price")
axes[0].set_title("🏠 Scikit-Learn Linear Regression")
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')

axes[1].scatter(y_test, y_pred_tf, color='dodgerblue', alpha=0.7, edgecolor='k')
axes[1].set_xlabel("Actual House Price")
axes[1].set_ylabel("Predicted House Price")
axes[1].set_title("🏠 TensorFlow Neural Network")
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')

plt.tight_layout()
plt.show()

# 3️⃣ Residual Plot (TensorFlow Model)
residuals_tf = y_test - y_pred_tf
residuals_sklearn = y_test - y_pred_sklearn

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(residuals_sklearn, bins=30, color='green', alpha=0.7, edgecolor='k')
axes[0].set_title("📉 Residuals - Scikit-Learn")
axes[0].set_xlabel("Residual Value")
axes[0].set_ylabel("Frequency")

axes[1].hist(residuals_tf, bins=30, color='purple', alpha=0.7, edgecolor='k')
axes[1].set_title("📉 Residuals - TensorFlow")
axes[1].set_xlabel("Residual Value")
axes[1].set_ylabel("Frequency")

plt.tight_layout()
plt.show()

# 4️⃣ Feature Importance (Coefficient Magnitude) - Scikit-Learn Model
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': sklearn_model.coef_
}).sort_values(by='Coefficient', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=coef_df, palette='viridis')
plt.title("📊 Feature Importance (Linear Regression Coefficients)")
plt.show()

# 5️⃣ Training History - TensorFlow Model
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['loss'], label='Training Loss')
axes[0].plot(history.history['val_loss'], label='Validation Loss')
axes[0].set_title("🔵 TensorFlow Model - Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid()

axes[1].plot(history.history['mae'], label='Training MAE')
axes[1].plot(history.history['val_mae'], label='Validation MAE')
axes[1].set_title("🔵 TensorFlow Model - MAE")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("MAE")
axes[1].legend()
axes[1].grid()

plt.tight_layout()
plt.show()








# "DM for alpha drops!" or "Follow for daily BTC briefs "

