import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load trained model
TRAINED_MODEL_PATH = './trained_models/custom_resnet50.hdf5'
model = load_model(TRAINED_MODEL_PATH)

# Load test data
test_data = pd.read_csv('./data/test.csv')
columns = ['EGFR']

# Test data generator
test_datagen = ImageDataGenerator(rescale=1. / 255.)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_data,
    x_col="filename",
    y_col=columns,
    batch_size=1,
    seed=42,
    shuffle=False,
    class_mode="raw",
    target_size=(256, 256)
)

# Predictions
predictions = model.predict(test_generator, verbose=1)
y_pred = (predictions > 0.5).astype(int)
y_true = test_data[columns].to_numpy()

# Metrics
def calc_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return acc, precision, recall, f1

metrics = calc_metrics(y_true, y_pred)
print("Accuracy:", metrics[0])
print("Precision:", metrics[1])
print("Recall:", metrics[2])
print("F1 Score:", metrics[3])
