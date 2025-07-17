import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import stats

# Load and clean data
df = pd.read_csv("12_FINAL_DataForTable1WithUpdatedVAriables28May2025.csv")
df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace(".", "", regex=False)

# Map AJCC stage
def map_stage(stage):
    stage = str(stage).strip().upper()
    if stage in ['I', 'IA', 'IB', 'IA1', 'IA2', 'IA3']:
        return 'I'
    elif stage in ['II', 'IIA', 'IIB']:
        return 'II'
    elif stage in ['III', 'IIIA', 'IIIB']:
        return 'III'
    elif stage in ['IV', 'IVA']:
        return 'IV'
    else:
        return 'Unknown'

df["STAGE_GROUPED"] = df["STAGE_COMPOSITE"].map(map_stage)

# Fill missing
df["SEX_DERIVED"] = df["SEX_DERIVED"].fillna("Unknown")
df["ASSIGNED_POPULATION"] = df["ASSIGNED_POPULATION"].fillna("Unknown")
df["GRADE_DIFFERENTIATION"] = df["GRADE_DIFFERENTIATION"].fillna("Unclassified")
df["STAGE_GROUPED"] = df["STAGE_GROUPED"].fillna("Unknown")

# Create STRATA key
df["STRATA"] = (
    df["SEX_DERIVED"].astype(str) + "_" +
    df["ASSIGNED_POPULATION"].astype(str) + "_" +
    df["STAGE_GROUPED"].astype(str) + "_" +
    df["GRADE_DIFFERENTIATION"].astype(str)
)

# Replace rare strata
rare_strata = df["STRATA"].value_counts()[df["STRATA"].value_counts() < 3].index
df["STRATA"] = df["STRATA"].apply(lambda x: "Other" if x in rare_strata else x)

# Stratified split
train_val_df, test_df = train_test_split(df, test_size=149, stratify=df["STRATA"], random_state=42)
assert len(train_val_df) == 346
assert len(test_df) == 149

# Columns
mutation_mt_cols = ["EGFR", "EGFR_pL858R", "EGFR_pE746_A750del", "TP53", "RBM10", "KRAS", "KRAS_pG12C", "KRAS_pG12D", "KRAS_pG12V"]
yes_cols = ["APOBEC_Signature", "CDKN2A_Del", "MDM2_Amp"]
y_cols = ["ALK_Fusion", "WGD_Status", "Kataegis"]

# Percent formatter
def pct(n, d):
    return f"{n} ({round((n / d) * 100)}%)" if d else f"{n} (0%)"

# Global p-value for categorical variable
# Global p-value for categorical variable
def pval_cat_global(train_col, test_col, use_fisher=False):
    table = pd.crosstab(train_col, np.repeat("Train", len(train_col))).join(
        pd.crosstab(test_col, np.repeat("Test", len(test_col))), how='outer'
    ).fillna(0)
    if use_fisher and table.shape == (2, 2):
        _, p = stats.fisher_exact(table.values)
    else:
        _, p, _, _ = stats.chi2_contingency(table)
    return f"{p:.2f}"

# t-test for continuous variable
def pval_cont(train, test):
    try:
        _, p = stats.ttest_ind(train, test, nan_policy='omit')
        return f"{p:.2f}"
    except Exception:
        return "NA"


# Age summary
def age_stats(s):
    return f"{int(s.mean())} ({int(s.median())}, {int(s.min())} - {int(s.max())})"

# Generic row
def make_row(category, total, train, test, pvalue):
    return {
        "Category": category,
        "Total": total,
        "Train/Validation": train,
        "p value": pvalue,
        "Testing": test
    }

# Breakdown for categorical var with global p-value
def cat_global_summary(col_name, label=None, use_fisher=False):
    label = label or col_name
    pvalue = pval_cat_global(train_val_df[col_name], test_df[col_name], use_fisher)
    rows = []
    for val in df[col_name].unique():
        total = pct((df[col_name] == val).sum(), len(df))
        train = pct((train_val_df[col_name] == val).sum(), len(train_val_df))
        test = pct((test_df[col_name] == val).sum(), len(test_df))
        rows.append(make_row(f"{label} = {val}", total, train, test, pvalue if rows == [] else ""))
    return rows

# Start table
table = []

# Age
age_p = pval_cont(train_val_df["AGE_AT_DIAGNOSIS"], test_df["AGE_AT_DIAGNOSIS"])
table.append({
    "Category": "Age at Diagnosis (years): Mean (Median, Range)",
    "Total": age_stats(df["AGE_AT_DIAGNOSIS"]),
    "Train/Validation": age_stats(train_val_df["AGE_AT_DIAGNOSIS"]),
    "p value": age_p,
    "Testing": age_stats(test_df["AGE_AT_DIAGNOSIS"])
})

# Demographic & clinical variables with global p-value
table += cat_global_summary("SEX_DERIVED", "Sex")
table += cat_global_summary("ASSIGNED_POPULATION", "WGS-based ancestry", use_fisher=True)
table += cat_global_summary("STAGE_GROUPED", "Stage Group")
table += cat_global_summary("GRADE_DIFFERENTIATION", "Grade")

# Genomic MT
for col in mutation_mt_cols:
    if col in df.columns:
        pvalue = pval_cat_global(train_val_df[col], test_df[col])
        total = pct((df[col] == "MT").sum(), len(df))
        train = pct((train_val_df[col] == "MT").sum(), len(train_val_df))
        test = pct((test_df[col] == "MT").sum(), len(test_df))
        table.append(make_row(f"{col} = MT", total, train, test, pvalue))

# Yes cols
for col in yes_cols:
    if col in df.columns:
        pvalue = pval_cat_global(train_val_df[col], test_df[col])
        total = pct((df[col] == "Yes").sum(), len(df))
        train = pct((train_val_df[col] == "Yes").sum(), len(train_val_df))
        test = pct((test_df[col] == "Yes").sum(), len(test_df))
        table.append(make_row(f"{col} = Yes", total, train, test, pvalue))

# Y/WGD cols
for col in y_cols:
    if col not in df.columns:
        continue
    val = "WGD" if col == "WGD_Status" else "Y"
    pvalue = pval_cat_global(train_val_df[col], test_df[col])
    total = pct((df[col] == val).sum(), len(df))
    train = pct((train_val_df[col] == val).sum(), len(train_val_df))
    test = pct((test_df[col] == val).sum(), len(test_df))
    table.append(make_row(f"{col} = {val}", total, train, test, pvalue))

# TMB
if "TMB" in df.columns:
    tmb = lambda s: f"{round(s.mean(), 1)} ({round(s.median(), 1)}, {int(s.min())} - {int(s.max())})"
    tmb_p = pval_cont(train_val_df["TMB"], test_df["TMB"])
    table.append({
        "Category": "TMB: Mean (Median, Range)",
        "Total": tmb(df["TMB"]),
        "Train/Validation": tmb(train_val_df["TMB"]),
        "p value": tmb_p,
        "Testing": tmb(test_df["TMB"])
    })

# Export
summary_df = pd.DataFrame(table)
summary_df = summary_df[["Category", "Total", "Train/Validation", "p value", "Testing"]]
summary_df.to_csv("Stratified_Final_Table_TrainVal_Test_with_Pvalue_3June2025.csv", index=False)
print("✅ Saved: Stratified_Final_Table_TrainVal_Test_with_Pvalue_3June2025.csv")
