import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import tensorflow_addons as tfa
from datetime import datetime
from model import build_custom_resnet50

# Hyperparameters
BATCH_SIZE = 64
TARGET_SIZE = (256, 256)
NUM_CLASSES = 16
EPOCHS = 200
LEARNING_RATE = 0.0001
SAVE_RESULTS_PATH = './results/'
TRAINED_MODEL_PATH = './trained_models/'

# Load dataset
columns = ['EGFR', 'KRAS', 'TP53', 'RBM10', 'EGFR_pL591R', 'EGFR_pE479_A483del', 'KRAS_pG12C',
           'KRAS_pG12V', 'KRAS_pG12D', 'CDKN2A_deletion', 'MDM2_amplification', 'ALK_fusion',
           'WGD', 'Kataegis', 'APOBEC', 'TMB']

data = pd.read_csv('./data/train.csv')

# Split data
df_train, df_val = train_test_split(data, train_size=0.75, random_state=42)

# Data Generators
data_gen_args = dict(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    zca_epsilon=1e-06,
    rescale=1. / 255.
)

train_datagen = ImageDataGenerator(**data_gen_args)
val_datagen = ImageDataGenerator(rescale=1. / 255.)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=df_train,
    x_col="filename",
    y_col=columns,
    batch_size=BATCH_SIZE,
    seed=42,
    shuffle=True,
    class_mode="raw",
    target_size=TARGET_SIZE
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=df_val,
    x_col="filename",
    y_col=columns,
    batch_size=BATCH_SIZE,
    seed=42,
    shuffle=True,
    class_mode="raw",
    target_size=TARGET_SIZE
)

# Build model
model = build_custom_resnet50((*TARGET_SIZE, 3), NUM_CLASSES)
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss='binary_crossentropy',
              metrics=[tfa.metrics.F1Score(num_classes=NUM_CLASSES, threshold=0.5, average='micro')])

# Callbacks
logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    TRAINED_MODEL_PATH + 'custom_resnet50.hdf5',
    monitor='val_f1_score',
    verbose=1,
    save_best_only=True
)
earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_f1_score', mode='max', patience=15)

# Training
history = model.fit(
    x=train_generator,
    steps_per_epoch=train_generator.n // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.n // val_generator.batch_size,
    epochs=EPOCHS,
    callbacks=[tensorboard_callback, model_checkpoint_callback, earlystop_callback]
)

# Save training history
history_df = pd.DataFrame(history.history)
history_df.to_csv(SAVE_RESULTS_PATH + 'training_history.csv', index=False)
