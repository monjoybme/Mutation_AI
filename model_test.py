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
columns = ['EGFR', 'KRAS', 'TP53', 'RBM10', 'EGFR_pL591R', 'EGFR_pE479_A483del', 'KRAS_pG12C',
           'KRAS_pG12V', 'KRAS_pG12D', 'CDKN2A_deletion', 'MDM2_amplification', 'ALK_fusion',
           'WGD', 'Kataegis', 'APOBEC', 'TMB']

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
    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average='macro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    micro_precision = precision_score(y_true, y_pred, average='micro')
    micro_recall = recall_score(y_true, y_pred, average='micro')
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    return acc, macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1

metrics = calc_metrics(y_true, y_pred)
print("Accuracy:", metrics[0])
print("Macro Precision:", metrics[1])
print("Macro Recall:", metrics[2])
print("Macro F1 Score:", metrics[3])
print("Micro Precision:", metrics[4])
print("Micro Recall:", metrics[5])
print("Micro F1 Score:", metrics[6])
