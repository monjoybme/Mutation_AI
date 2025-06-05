"""
Description:
------------
This script performs a statistical comparison of the AUCs (Area Under the ROC Curve)
from two classification models evaluated on the same dataset, using the DeLong test.

Key Features:
-------------
- Computes ROC AUC scores for two sets of model predictions.
- Implements DeLong's test to assess if the difference in AUCs is statistically significant.
  (This test correctly accounts for the covariance between model predictions, 
   i.e., the variance of the difference is NOT the sum of variances.)
- Applies Benjamini–Hochberg (FDR) correction to the resulting p-value for multiple testing.
- Outputs AUCs, raw p-value, corrected p-value, and significance decision to a CSV file.

Input:
------
CSV file named 'your_data.csv' with the following columns:
  - true_labels: Ground truth binary labels (0 or 1).
  - model1_predictions: Predicted probabilities from model 1.
  - model2_predictions: Predicted probabilities from model 2.

Output:
-------
CSV file named 'delong_test_results.csv' with:
  - AUCs for both models
  - Raw DeLong p-value
  - Corrected p-value (Benjamini–Hochberg)
  - Whether the result is statistically significant after correction

Reference:
----------
DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988). 
Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach.

Custom DeLong implementation adapted from:
https://github.com/yandexdataschool/roc_comparison/blob/master/compare_auc_delong_xu.py
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

# Custom DeLong implementation (from delong.py or similar)
# Source: https://github.com/yandexdataschool/roc_comparison/blob/master/compare_auc_delong_xu.py
import scipy.stats

def compute_midrank(x):
    """Computes midranks."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2

def compute_ground_truth_statistics(ground_truth):
    order = np.argsort(-ground_truth)
    label_1_count = np.sum(ground_truth)
    label_0_count = ground_truth.shape[0] - label_1_count
    return label_1_count, label_0_count

def fastDeLong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m])
    ty = np.empty([k, n])
    tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = (tx.sum(axis=1) / m - (m + 1) / 2) / n

    v01 = (tz[:, :m] - tx) / n
    v10 = (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    s = sx / m + sy / n
    return aucs, s

def calc_pvalue(aucs, sigma):
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    pvalue = 2 * (1 - scipy.stats.norm.cdf(z))
    return pvalue

def delong_roc_test(ground_truth, predictions_one, predictions_two):
    """Performs DeLong test for two correlated ROC AUCs."""
    predictions = np.array([predictions_one, predictions_two])
    label_1_count = int(np.sum(ground_truth))
    aucs, covariance = fastDeLong(predictions, label_1_count)
    pvalue = float(calc_pvalue(aucs, covariance))
    return aucs, pvalue

# --- MAIN EXECUTION ---

# 1. Load CSV
df = pd.read_csv("your_data.csv")

# 2. Extract columns
true_labels = df['true_labels'].values
pred1 = df['model1_predictions'].values
pred2 = df['model2_predictions'].values

# 3. Compute AUCs
auc1 = roc_auc_score(true_labels, pred1)
auc2 = roc_auc_score(true_labels, pred2)

# 4. Run DeLong Test
aucs, pval = delong_roc_test(true_labels, pred1, pred2)

# 5. Multiple testing correction (Benjamini-Hochberg)
pvals = np.array([pval])  # In real case, you might have more tests
rej, corrected_pvals, _, _ = multipletests(pvals, method='fdr_bh')

# 6. Save results
results_df = pd.DataFrame({
    "Model 1 AUC": [auc1],
    "Model 2 AUC": [auc2],
    "Raw p-value": [pval],
    "Corrected p-value (BH)": corrected_pvals,
    "Significant (BH)": rej
})

results_df.to_csv("delong_test_results.csv", index=False)

print("Results saved to 'delong_test_results.csv'")
