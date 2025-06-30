from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

def calc_metrics(y_true, y_pred, average='macro'):
    """
    Calculate various metrics including accuracy, precision, recall, F1, sensitivity, and specificity.

    Args:
        y_true (np.array): Ground truth binary labels (multi-label).
        y_pred (np.array): Predicted binary labels.
        average (str): averaging method for precision, recall, f1 ('macro', 'micro', 'weighted').

    Returns:
        dict: A dictionary containing all the calculated metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    micro_precision = precision_score(y_true, y_pred, average='micro', zero_division=0)
    micro_recall = recall_score(y_true, y_pred, average='micro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)

    # Sensitivity = recall (same as above)
    sensitivity = macro_recall

    # Specificity calculation per label and averaged
    specificities = []
    for i in range(y_true.shape[1]):
        tn, fp, fn, tp = confusion_matrix(y_true[:, i], y_pred[:, i], labels=[0, 1]).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificities.append(spec)
    specificity = np.mean(specificities)

    return {
        "accuracy": acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "sensitivity": sensitivity,
        "specificity": specificity
    }
