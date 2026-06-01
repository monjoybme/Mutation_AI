import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import build_custom_resnet50

# Paths and parameters
TEST_DATA_CSV = 'Final_VALIDATIONData_14March2025.csv'
MODEL_PATH = './trained_models/custom_resnet50.keras'
RESULTS_PATH = './results/'
TARGET_SIZE = (256, 256)
BATCH_SIZE = 64
COLUMN = 'EGFR'

# Load test data
test_data = pd.read_csv(TEST_DATA_CSV)

# Data generator for test (only rescale)
test_datagen = ImageDataGenerator(rescale=1. / 255.)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_data,
    x_col='filename',
    y_col=[COLUMN],
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_mode='raw',
    target_size=TARGET_SIZE
)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded for inference.")

# Predict probabilities
y_true = test_generator.labels
y_pred_probs = model.predict(test_generator, steps=test_generator.n // test_generator.batch_size + 1)
y_pred_probs = y_pred_probs.flatten()  # shape (N,)

# Load or define threshold (you can tune this with validation data, here we pick 0.5 default)
threshold = 0.5

# Apply threshold
y_pred_bin = (y_pred_probs >= threshold).astype(int)

# Metrics
roc_auc = roc_auc_score(y_true, y_pred_probs)

tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin).ravel()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

print(f"ROC AUC: {roc_auc:.4f}")
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")

print("\nClassification Report:")
print(classification_report(y_true, y_pred_bin, zero_division=0))

# Save predictions and metrics
results_df = pd.DataFrame({
    'filename': test_generator.filenames,
    'true_label': y_true,
    'pred_prob': y_pred_probs,
    'pred_label': y_pred_bin
})
results_df.to_csv(RESULTS_PATH + 'test_predictions.csv', index=False)

metrics_summary = {
    'roc_auc': [roc_auc],
    'sensitivity': [sensitivity],
    'specificity': [specificity]
}
metrics_df = pd.DataFrame(metrics_summary)
metrics_df.to_csv(RESULTS_PATH + 'test_metrics.csv', index=False)

print(f"Test predictions saved to {RESULTS_PATH}test_predictions.csv")
print(f"Test metrics saved to {RESULTS_PATH}test_metrics.csv")
