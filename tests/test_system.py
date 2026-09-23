from src.analysis import run_analysis


def test_complete_analysis_workflow(tmp_path, sample_data):
    """System test: run the workflow from a CSV file through model evaluation."""
    file_path = tmp_path / "lung_cancer_dataset.csv"
    sample_data.to_csv(file_path, index=False)

    results = run_analysis(file_path)

    assert len(results["data"]) == len(sample_data)
    assert results["stage_summary"]["Patient_Count"].sum() == len(sample_data)
    assert 0 <= results["accuracy"] <= 1

