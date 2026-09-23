"""Core data-processing and modeling functions used by the notebook."""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


REQUIRED_COLUMNS = {
    "Patient_ID",
    "Age",
    "Smoking_Status",
    "Cancer_Stage",
    "Tumor_Size_cm",
    "Survival_Months",
    "Survived",
}

STAGE_MAP = {
    "Stage I": 1,
    "Stage II": 2,
    "Stage III": 3,
    "Stage IV": 4,
}


def load_data(file_path):
    """Load the CSV file and confirm that the analysis columns exist."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(file_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    return df


def preprocess_data(df):
    """Create numeric survival and cancer-stage columns for analysis."""
    processed = df.copy()
    processed["Survived_num"] = processed["Survived"].map({"Yes": 1, "No": 0})
    processed["Stage_num"] = processed["Cancer_Stage"].map(STAGE_MAP)

    if processed[["Survived_num", "Stage_num"]].isna().any().any():
        raise ValueError("Survived or Cancer_Stage contains an unexpected value.")
    return processed


def calculate_stage_survival(df):
    """Return survival percentage and patient count for each cancer stage."""
    summary = df.groupby("Cancer_Stage")["Survived_num"].agg(["mean", "count"])
    summary["mean"] = summary["mean"] * 100
    return summary.rename(
        columns={"mean": "Survival_Rate", "count": "Patient_Count"}
    )


def filter_stages(df, stages=("Stage I", "Stage IV")):
    """Keep rows belonging to the requested cancer stages."""
    return df[df["Cancer_Stage"].isin(stages)].copy()


def smoking_counts_by_stage(df):
    """Count smoking-status categories within each cancer stage."""
    return df.groupby(["Cancer_Stage", "Smoking_Status"]).size().unstack(fill_value=0)


def train_and_evaluate_model(df, test_size=0.2, random_state=42):
    """Train logistic regression and return the model, predictions, and accuracy."""
    features = ["Stage_num", "Tumor_Size_cm", "Age"]
    X = df[features]
    y = df["Survived_num"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return model, predictions, accuracy, X_train, X_test, y_train, y_test


def run_analysis(file_path):
    """Run the main workflow from data loading through model evaluation."""
    raw_df = load_data(file_path)
    processed_df = preprocess_data(raw_df)
    stage_summary = calculate_stage_survival(processed_df)
    model_results = train_and_evaluate_model(processed_df)

    return {
        "data": processed_df,
        "stage_summary": stage_summary,
        "model": model_results[0],
        "accuracy": model_results[2],
    }

