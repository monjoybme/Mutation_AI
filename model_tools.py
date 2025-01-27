from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calc_metrics(y_true, y_pred):
    """
    Calculate various metrics including accuracy, precision, recall, and F1 scores.

    Args:
        y_true (np.array): Ground truth labels.
        y_pred (np.array): Predicted labels.

    Returns:
        dict: A dictionary containing all the calculated metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average='macro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    micro_precision = precision_score(y_true, y_pred, average='micro')
    micro_recall = recall_score(y_true, y_pred, average='micro')
    micro_f1 = f1_score(y_true, y_pred, average='micro')

    return {
        "accuracy": acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1
    }
