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

