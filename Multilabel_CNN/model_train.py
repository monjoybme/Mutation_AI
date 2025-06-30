import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from datetime import datetime
from model import build_custom_resnet50
import os

# --------------------------
# Hyperparameters
# --------------------------
BATCH_SIZE = 64
TARGET_SIZE = (256, 256)
NUM_CLASSES = 16
EPOCHS = 200
LEARNING_RATE = 0.0001
SAVE_RESULTS_PATH = './results/'
TRAINED_MODEL_PATH = './trained_models/custom_resnet50.keras'

# --------------------------
# Load dataset
# --------------------------
columns = ['EGFR', 'KRAS', 'TP53', 'RBM10', 'EGFR_p858R', 'EGFR_E746_A750del', 'KRAS_pG12C',
           'KRAS_pG12V', 'KRAS_pG12D', 'CDKN2A_deletion', 'MDM2_amplification', 'ALK_fusion',
           'WGD', 'Kataegis', 'APOBEC', 'TMB']

data = pd.read_csv('Final_TRAININGData_14March2025.csv')

# --------------------------
# Split data
# --------------------------
df_train, df_val = train_test_split(data, train_size=0.9, random_state=42)

# --------------------------
# Compute class weights
# --------------------------
label_counts = df_train[columns].sum()
total_samples = len(df_train)
epsilon = 1e-6
class_weights = total_samples / (len(columns) * (label_counts + epsilon))
class_weights = class_weights / np.sum(class_weights) * len(columns)
class_weights_tensor = tf.constant(class_weights.values, dtype=tf.float32)

print("Per-label class weights:")
for label, weight in zip(columns, class_weights):
    print(f"{label}: {weight:.4f}")

# --------------------------
# Custom loss function
# --------------------------
def weighted_bce(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    loss = - (class_weights_tensor * (y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred)))
    return tf.reduce_mean(loss)

# --------------------------
# Image processing function
# --------------------------
def load_and_preprocess(filename, label):
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, TARGET_SIZE)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_rotation(img, 20)
    img = tf.image.random_brightness(img, 0.2)
    img = img / 255.0
    return img, label

# --------------------------
# Balanced batch sampler dataset
# --------------------------
def create_balanced_dataset(df, batch_size):
    # Oversample minority labels: sample rows with rare labels more often
    # Compute sample weights = max label weight of the row
    sample_weights = df[columns].dot(class_weights)
    sample_probs = sample_weights / sample_weights.sum()
    
    filepaths = df['filename'].values
    labels = df[columns].values.astype(np.float32)

    def gen():
        while True:
            indices = np.random.choice(len(df), size=batch_size, p=sample_probs)
            for idx in indices:
                yield filepaths[idx], labels[idx]
    
    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(NUM_CLASSES,), dtype=tf.float32)
        )
    )
    ds = ds.map(lambda x, y: load_and_preprocess(x, y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = create_balanced_dataset(df_train, BATCH_SIZE)

# Validation dataset (no augmentation)
def load_val(filename, label):
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, TARGET_SIZE)
    img = img / 255.0
    return img, label

val_ds = tf.data.Dataset.from_tensor_slices((df_val['filename'].values, df_val[columns].values.astype(np.float32)))
val_ds = val_ds.map(load_val, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# --------------------------
# Build or load model
# --------------------------
if os.path.exists(TRAINED_MODEL_PATH):
    model = tf.keras.models.load_model(TRAINED_MODEL_PATH, compile=False)
    print("Loaded model from checkpoint.")
else:
    model = build_custom_resnet50((*TARGET_SIZE, 3), NUM_CLASSES)
    print("Created a new model.")

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss=weighted_bce,
    metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# --------------------------
# Callbacks
# --------------------------
logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    TRAINED_MODEL_PATH,
    monitor='val_binary_accuracy',
    verbose=1,
    save_best_only=True
)
earlystop_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_binary_accuracy',
    mode='max',
    patience=15
)

# --------------------------
# Train
# --------------------------
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=[tensorboard_callback, model_checkpoint_callback, earlystop_callback]
)

# Save training history
os.makedirs(SAVE_RESULTS_PATH, exist_ok=True)
pd.DataFrame(history.history).to_csv(os.path.join(SAVE_RESULTS_PATH, 'training_history.csv'), index=False)

# --------------------------
# Threshold tuning with Youden’s index
# --------------------------
# Get validation predictions
y_true = np.vstack([y for _, y in val_ds])
y_pred = model.predict(val_ds)

# Find optimal thresholds
optimal_thresholds = []
for i in range(NUM_CLASSES):
    fpr, tpr, thresholds = roc_curve(y_true[:, i], y_pred[:, i])
    youden_index = tpr - fpr
    best_idx = np.argmax(youden_index)
    optimal_thresholds.append(thresholds[best_idx])
    print(f"{columns[i]}: Optimal threshold = {thresholds[best_idx]:.4f}, Youden’s index = {youden_index[best_idx]:.4f}")

# Save thresholds
threshold_df = pd.DataFrame({
    'Label': columns,
    'OptimalThreshold': optimal_thresholds
})
threshold_df.to_csv(os.path.join(SAVE_RESULTS_PATH, 'optimal_thresholds.csv'), index=False)
