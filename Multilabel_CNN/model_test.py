import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import tensorflow as tf

# --------------------------
# Load trained model
# --------------------------
model = tf.keras.models.load_model('./trained_models/custom_resnet50.keras', compile=False)

# --------------------------
# Load thresholds
# --------------------------
threshold_df = pd.read_csv('./results/optimal_thresholds.csv')
thresholds_array = threshold_df['OptimalThreshold'].values

# --------------------------
# Load validation data (or test data if available)
# --------------------------
columns = ['EGFR', 'KRAS', 'TP53', 'RBM10', 'EGFR_p858R', 'EGFR_E746_A750del', 'KRAS_pG12C',
           'KRAS_pG12V', 'KRAS_pG12D', 'CDKN2A_deletion', 'MDM2_amplification', 'ALK_fusion',
           'WGD', 'Kataegis', 'APOBEC', 'TMB']

val_data = pd.read_csv('Final_VALIDATIONData_14March2025.csv')  # Adjust if using separate test data

TARGET_SIZE = (256, 256)

def load_val(filename, label):
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, TARGET_SIZE)
    img = img / 255.0
    return img, label

val_ds = tf.data.Dataset.from_tensor_slices((
    val_data['filename'].values,
    val_data[columns].values.astype(np.float32)
))
val_ds = val_ds.map(load_val, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.batch(64).prefetch(tf.data.AUTOTUNE)

# --------------------------
# Run inference
# --------------------------
y_true = np.vstack([y for _, y in val_ds])
y_pred_probs = model.predict(val_ds)

# Apply thresholds
y_pred_bin = (y_pred_probs >= thresholds_array).astype(int)

# --------------------------
# Calculate ROC AUC, Sensitivity, Specificity per label
# --------------------------
metrics = {
    'Label': [],
    'ROC_AUC': [],
    'Sensitivity': [],   # Recall: TP/(TP+FN)
    'Specificity': []    # TN/(TN+FP)
}

for i, col in enumerate(columns):
    metrics['Label'].append(col)
    # ROC AUC
    try:
        auc = roc_auc_score(y_true[:, i], y_pred_probs[:, i])
    except ValueError:
        auc = np.nan  # Only one class present in y_true[:, i]
    metrics['ROC_AUC'].append(auc)

    # Confusion matrix components for binary classification
    tn, fp, fn, tp = confusion_matrix(y_true[:, i], y_pred_bin[:, i], labels=[0,1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    metrics['Sensitivity'].append(sensitivity)
    metrics['Specificity'].append(specificity)

    print(f"{col}: ROC AUC={auc:.4f}, Sensitivity={sensitivity:.4f}, Specificity={specificity:.4f}")

# Save metrics
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv('./results/inference_metrics.csv', index=False)

# --------------------------
# Optional: Detailed classification report
# --------------------------
report = classification_report(y_true, y_pred_bin, target_names=columns, zero_division=0, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv('./results/classification_report.csv')

print("\nMetrics saved to ./results/inference_metrics.csv")
print("Classification report saved to ./results/classification_report.csv")
