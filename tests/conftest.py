import pandas as pd
import pytest


@pytest.fixture
def sample_data():
    """Small, balanced dataset used by the automated tests."""
    rows = []
    stages = ["Stage I", "Stage II", "Stage III", "Stage IV"]

    for i in range(40):
        stage = stages[i % 4]
        rows.append(
            {
                "Patient_ID": f"LC-{i:04d}",
                "Age": 40 + i,
                "Smoking_Status": "Never Smoked" if i % 2 == 0 else "Current Smoker",
                "Cancer_Stage": stage,
                "Tumor_Size_cm": 1.0 + (i % 8),
                "Survival_Months": 80 - i,
                "Survived": "Yes" if i % 2 == 0 else "No",
            }
        )

    return pd.DataFrame(rows)

