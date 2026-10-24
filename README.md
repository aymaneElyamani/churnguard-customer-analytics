# ChurnGuard — Customer Churn Analytics & Prediction

ChurnGuard is an end-to-end Data Science and Machine Learning project built to analyze customer purchasing behavior and estimate the likelihood of an observed future purchase using the Brazilian Olist e-commerce dataset.

The project combines customer analytics, feature engineering, machine learning, a REST API, an interactive dashboard, and containerized deployment.

## Project Objectives

- Analyze customer purchasing behavior and retention patterns.
- Perform RFM segmentation (Recency, Frequency, Monetary Value).
- Explore customer cohorts and repeat-purchase behavior.
- Build customer-level features from historical delivered orders.
- Investigate the challenges of predicting rare repeat purchases.
- Train a classification model and expose predictions through a REST API.
- Visualize customer analytics through an interactive dashboard.
- Package the application using Docker and Docker Compose.

## Technology Stack

| Area              | Technologies               |
| ----------------- | -------------------------- |
| Programming       | Python                     |
| Data analysis     | pandas, NumPy              |
| Machine Learning  | scikit-learn               |
| API               | FastAPI, Uvicorn           |
| Dashboard         | Streamlit                  |
| Model persistence | joblib                     |
| Testing           | pytest, FastAPI TestClient |
| Deployment        | Docker, Docker Compose     |
| Version control   | Git, GitHub                |

## Architecture

```text
Olist Dataset
     |
     v
Data Preparation & Feature Engineering
     |
     v
Customer Analytics & Model Training
     |
     v
Saved Model + Metadata
     |
     v
FastAPI Prediction Service
     |
     v
Streamlit Dashboard
```

## Main Components

### Customer Analytics

- RFM customer segmentation.
- Cohort retention analysis.
- Repeat-purchase behavior analysis.
- Feature comparison and temporal validation experiments.
- Data quality and target-definition investigations.

### Machine Learning

The project uses a Logistic Regression pipeline with feature scaling and class weighting to address the highly imbalanced target.

The final training dataset contains 25,926 customer records, including 356 observed future purchases (approximately 1.37%).

The selected features are:

- `frequence_180j`: number of delivered orders during the observation window.
- `montant_articles_180j`: total item amount during the observation window.
- `recence_jours`: number of days since the most recent delivered order.

### REST API

The FastAPI application exposes:

- `GET /` — basic service information.
- `GET /health` — health check.
- `POST /predict` — prediction endpoint.
- `/docs` — interactive API documentation.

### Interactive Dashboard

The Streamlit dashboard provides an interface for interacting with the prediction API.

### Docker Deployment

Docker Compose runs two services:

- **API:** FastAPI on port `8000`.
- **Dashboard:** Streamlit exposed on port `18501`.

## Project Structure

```text
churnguard-customer-analytics/
├── api.py
├── dashboard.py
├── train_final_model.py
├── requirements.txt
├── Dockerfile
├── compose.yaml
├── models/
│   ├── churnguard_model_v1.joblib
│   └── model_metadata.json
├── tests/
│   └── test_api.py
├── reports/
│   └── figures/
├── data/
│   ├── raw/
│   └── processed/
├── README.md
├── .gitignore
└── .dockerignore
```

The repository also includes analysis, data preparation, and model evaluation scripts documenting the project's experiments. Raw and processed CSV datasets are kept locally and excluded from Git.

## Getting Started

### Prerequisites

- Python 3.12
- Git
- Docker Desktop with Docker Compose

### Run with Docker Compose

Clone the repository:

```bash
git clone https://github.com/aymaneElyamani/churnguard-customer-analytics.git
cd churnguard-customer-analytics
```

Build and start the services:

```bash
docker compose up --build -d
```

Check the service status:

```bash
docker compose ps
```

Access the application:

- Dashboard: [http://localhost:18501](http://localhost:18501)
- API documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health endpoint: [http://localhost:8000/health](http://localhost:8000/health)

Stop the services:

```bash
docker compose down
```

### Run Tests Locally

Create and activate a virtual environment on Windows:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```cmd
python -m pip install -r requirements.txt
```

Run the tests from the project root:

```cmd
python -m pytest -v
```

## Dataset

The project uses the Brazilian E-Commerce Public Dataset by Olist.

The original CSV files should be placed in `data/raw/` before running data preparation scripts. Generated datasets belong in `data/processed/`.

These data directories are excluded from Git to avoid committing large data files. Refer to the dataset publisher's terms when obtaining and using the data.

## Key Findings and Limitations

- Repeat purchases are relatively rare in the evaluated customer population.
- The target is defined using an observed future delivered purchase within a specified prediction window.
- No observed purchase during that window does **not** prove permanent customer churn.
- Strong class imbalance makes accuracy alone an unsuitable measure of model quality.
- Model performance must be assessed using appropriate metrics, such as average precision, precision, recall, and lift at a selected customer ranking cutoff.
- Temporal validation and possible customer overlap must be considered when interpreting evaluation results.

ChurnGuard is a portfolio and learning project. Its model is not yet validated for production use, and its predictions should not be interpreted as definitive statements about customer behavior.

## Future Improvements

- Strengthen leakage prevention and temporal validation.
- Improve model evaluation and probability calibration.
- Compare additional classification models and decision thresholds.
- Expand automated data quality checks.
- Improve dashboard analytics and model monitoring.
- Develop a reproducible end-to-end training pipeline.

## Author

**Aymane El Yamani**

Data Science & AI | Software Engineering

GitHub: [aymaneElyamani](https://github.com/aymaneElyamani)
