from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calc_metrics(y_true, y_pred):
    """
    Calculate binary classification metrics.

    Args:
        y_true (np.array): Ground truth labels.
        y_pred (np.array): Predicted labels.

    Returns:
        dict: A dictionary containing accuracy, precision, recall, and F1 score.
    """
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
