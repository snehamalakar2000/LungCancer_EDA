FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY tests/ tests/
COPY lung_cancer_dataset.csv .

CMD ["python", "-c", "from src.analysis import run_analysis; results = run_analysis('lung_cancer_dataset.csv'); print(results['stage_summary']); print('Model accuracy: {:.2%}'.format(results['accuracy']))"]