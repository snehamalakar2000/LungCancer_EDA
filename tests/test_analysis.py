import pandas as pd
import pytest

from src.analysis import (
    calculate_stage_survival,
    filter_stages,
    load_data,
    preprocess_data,
    smoking_counts_by_stage,
    train_and_evaluate_model,
)


def test_load_data_reads_valid_csv(tmp_path, sample_data):
    """A valid CSV should load with the same rows and columns."""
    file_path = tmp_path / "sample.csv"
    sample_data.to_csv(file_path, index=False)

    loaded = load_data(file_path)

    pd.testing.assert_frame_equal(loaded, sample_data)


def test_load_data_rejects_missing_columns(tmp_path):
    """The loader should clearly reject a dataset missing required columns."""
    file_path = tmp_path / "incomplete.csv"
    pd.DataFrame({"Patient_ID": ["LC-0001"]}).to_csv(file_path, index=False)

    with pytest.raises(ValueError, match="missing columns"):
        load_data(file_path)


def test_preprocess_data_encodes_categories(sample_data):
    """Yes/No and stage labels should be converted to the expected numbers."""
    processed = preprocess_data(sample_data)

    assert set(processed["Survived_num"].unique()) == {0, 1}
    assert set(processed["Stage_num"].unique()) == {1, 2, 3, 4}
    assert "Survived_num" not in sample_data.columns


def test_stage_survival_calculation(sample_data):
    """The stage summary should report percentages and correct patient counts."""
    processed = preprocess_data(sample_data)
    summary = calculate_stage_survival(processed)

    assert list(summary.columns) == ["Survival_Rate", "Patient_Count"]
    assert summary["Patient_Count"].sum() == len(sample_data)
    assert summary["Survival_Rate"].between(0, 100).all()


def test_stage_filter_and_smoking_counts(sample_data):
    """Filtering should retain only selected stages and preserve all their rows."""
    filtered = filter_stages(sample_data)
    counts = smoking_counts_by_stage(filtered)

    assert set(filtered["Cancer_Stage"]) == {"Stage I", "Stage IV"}
    assert counts.to_numpy().sum() == len(filtered)


def test_model_training_returns_valid_results(sample_data):
    """Model training should produce one prediction per test row and valid accuracy."""
    processed = preprocess_data(sample_data)
    model, predictions, accuracy, _, X_test, _, _ = train_and_evaluate_model(processed)

    assert hasattr(model, "predict")
    assert len(predictions) == len(X_test)
    assert 0 <= accuracy <= 1


def test_load_data_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        load_data(tmp_path / "missing.csv")


def test_load_data_rejects_empty_dataset(tmp_path, sample_data):
    file_path = tmp_path / "empty.csv"
    sample_data.iloc[:0].to_csv(file_path, index=False)
    with pytest.raises(ValueError, match="Dataset is empty"):
        load_data(file_path)


@pytest.mark.parametrize("column", ["Age", "Survived", "Cancer_Stage"])
def test_preprocess_rejects_missing_values(sample_data, column):
    invalid = sample_data.copy()
    invalid[column] = invalid[column].astype(object)
    invalid.loc[0, column] = None
    with pytest.raises(ValueError, match="Missing values"):
        preprocess_data(invalid)


@pytest.mark.parametrize(
    "column,value",
    [("Survived", "Maybe"), ("Cancer_Stage", "Stage V")],
)
def test_preprocess_rejects_unknown_labels(sample_data, column, value):
    invalid = sample_data.copy()
    invalid.loc[0, column] = value
    with pytest.raises(ValueError, match="unexpected value"):
        preprocess_data(invalid)


@pytest.mark.parametrize("value", ["unknown", float("inf"), -1])
def test_preprocess_rejects_invalid_tumor_size(sample_data, value):
    invalid = sample_data.copy()
    invalid["Tumor_Size_cm"] = invalid["Tumor_Size_cm"].astype(object)
    invalid.loc[0, "Tumor_Size_cm"] = value
    with pytest.raises(ValueError, match="Tumor_Size_cm"):
        preprocess_data(invalid)


def test_stage_survival_with_known_answer(sample_data):
    """Two patients with one survivor must produce exactly 50% survival."""
    small = sample_data.iloc[:2].copy()
    small["Cancer_Stage"] = "Stage I"
    small["Survived"] = ["Yes", "No"]
    summary = calculate_stage_survival(preprocess_data(small))
    assert summary.loc["Stage I", "Survival_Rate"] == 50
    assert summary.loc["Stage I", "Patient_Count"] == 2


def test_filter_with_no_matching_stage(sample_data):
    assert filter_stages(sample_data, stages=("Stage V",)).empty
