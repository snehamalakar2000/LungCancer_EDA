# Lung Cancer EDA and Survival Prediction

[![Python tests](https://github.com/snehamalakar2000/LungCancer_EDA/actions/workflows/tests.yml/badge.svg)](https://github.com/snehamalakar2000/LungCancer_EDA/actions/workflows/tests.yml)

This project explores a lung cancer dataset and trains a logistic regression model to predict whether a patient survived. It also includes automated unit and system tests so the analysis is reproducible and reliable.

## Project structure

```text
.
├── .github/workflows/tests.yml  # Runs tests automatically on GitHub
├── src/analysis.py              # Reusable analysis functions
├── tests/test_analysis.py       # Unit tests
├── tests/test_system.py         # End-to-end system test
├── EDA.ipynb            # Exploratory analysis and model
├── lung_cancer_dataset.csv      # Dataset (add this file locally)
└── requirements.txt             # Python dependencies
```

## Analysis

The notebook:

1. Loads and inspects the data.
2. Converts survival and cancer-stage labels to numeric values.
3. Compares survival rates across cancer stages.
4. Examines smoking status for Stage I and Stage IV patients.
5. Measures the relationship between tumor size and survival time.
6. Trains and evaluates a logistic regression model using cancer stage, tumor size, and age.

## Run the project

Place `lung_cancer_dataset.csv` in the project root. Then install the dependencies and open the notebook:

```bash
pip install -r requirements.txt
jupyter notebook EDA.ipynb
```

Run all notebook cells from top to bottom.

## Run the tests

From the project root, run:

```bash
pytest -v
```

The test suite includes checks for data loading, missing columns, preprocessing, stage summaries, filtering, model training, and the complete analysis workflow.

## Continuous integration

The GitHub Actions workflow runs the full test suite whenever code is pushed or a pull request is opened. A successful run confirms that all automated tests pass in a clean Python environment.

Before submitting, replace `USERNAME/REPOSITORY` in the badge above with your GitHub username and repository name. Then add a screenshot of the successful test run below.

## Test results

Add your screenshot here after GitHub Actions passes:

```markdown
![Successful test run](images/tests-passed.png)
```

