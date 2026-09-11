# Credit Card Fraud Detection

An end-to-end machine learning project for detecting fraudulent credit card transactions. The project covers model experimentation, model comparison, hyperparameter tuning, final model training, REST API development using FastAPI, Docker containerization, and cloud deployment.

## Project Overview

Credit card fraud detection is a binary classification problem where each transaction is classified as either:

* `0` — Legitimate transaction
* `1` — Fraudulent transaction

The dataset is highly imbalanced, making fraud detection more challenging. Because of this, model evaluation is not based on accuracy alone. Recall, F1-score, ROC-AUC, and PR-AUC are also considered when selecting the final model.

## Project Workflow

```text
Dataset
   |
   v
Exploratory Data Analysis
   |
   v
Data Preparation
   |
   v
Model Comparison
   |
   v
Hyperparameter Tuning
   |
   v
Final Model Training
   |
   v
Model Serialization
   |
   v
FastAPI REST API
   |
   v
Docker Containerization
   |
   v
Cloud Deployment
```

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Random Forest
* Joblib
* FastAPI
* Pydantic
* Uvicorn
* Docker
* Git
* GitHub
* Render

## Project Structure

```text
credit_card_resume/
|
├── data/
│   └── creditcard.csv
|
├── notebooks/
│   ├── Choosing_model.ipynb
│   └── fine_tune.ipynb
|
├── src/
│   ├── app.py
│   └── training.py
|
├── .dockerignore
├── .gitignore
├── Dockerfile
├── model.pkl
├── README.md
└── requirements.txt
```

### File Description

| File/Folder                      | Description                                  |
| -------------------------------- | -------------------------------------------- |
| `data/`                          | Contains the credit card transaction dataset |
| `notebooks/Choosing_model.ipynb` | Model comparison and evaluation              |
| `notebooks/fine_tune.ipynb`      | Hyperparameter tuning                        |
| `src/training.py`                | Trains the final Random Forest model         |
| `src/app.py`                     | FastAPI application for serving predictions  |
| `model.pkl`                      | Serialized trained Random Forest model       |
| `Dockerfile`                     | Configuration for building the Docker image  |
| `requirements.txt`               | Python dependencies                          |
| `README.md`                      | Project documentation                        |

## Dataset

The dataset contains credit card transactions with anonymized features.

The target variable is:

```text
Class
```

where:

```text
0 = Legitimate
1 = Fraudulent
```

The model uses the following input features:

```text
Time
V1
V2
V3
V4
V5
V6
V7
V8
V9
V10
V11
V12
V13
V14
V15
V16
V17
V18
V19
V20
V21
V22
V23
V24
V25
V26
V27
V28
Amount
```

`V1` through `V28` are anonymized features provided by the dataset.

## Model Comparison

Multiple machine learning models were evaluated to determine which model performed best for the fraud detection task.

The following evaluation metrics were used:

* Recall
* F1-score
* Accuracy
* ROC-AUC
* PR-AUC

### Results

| Model               | Recall | F1-Score | Accuracy | ROC-AUC | PR-AUC |
| ------------------- | -----: | -------: | -------: | ------: | -----: |
| Logistic Regression | 0.6268 |   0.7286 |   0.9992 |  0.9793 | 0.7598 |
| Decision Tree       | 0.7283 |   0.7333 |   0.9991 |  0.8639 | 0.5394 |
| Random Forest       | 0.7588 |   0.8384 |   0.9995 |  0.9450 | 0.8386 |
| XGBoost             | 0.5993 |   0.6465 |   0.9987 |  0.8452 | 0.6108 |

## Model Selection

Based on the comparison, Random Forest was selected as the final model.

Random Forest achieved the highest:

* Recall: `0.7588`
* F1-score: `0.8384`
* Accuracy: `0.9995`
* PR-AUC: `0.8386`

Logistic Regression achieved the highest ROC-AUC of `0.9793`, but Random Forest provided a better overall balance of recall, F1-score, and PR-AUC.

Since this is a fraud detection problem with an imbalanced target variable, recall and PR-AUC are particularly important. A model that has high accuracy but fails to identify fraudulent transactions would not be suitable for this application.

Therefore, Random Forest was chosen as the final model.

## Final Model

The final model is a Random Forest Classifier with the following configuration:

```python
RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    max_depth=None
)
```

The model is trained on the available dataset and saved using Joblib:

```python
joblib.dump(model, MODEL_PATH)
```

The resulting model is stored as:

```text
model.pkl
```

## FastAPI

The trained model is served through a REST API using FastAPI.

### API Endpoints

#### Health Check

```http
GET /
```

Response:

```json
{
    "message": "Credit Card Fraud Detection API is running"
}
```

#### Fraud Prediction

```http
POST /predict
```

The endpoint accepts transaction features and returns the predicted class, fraud probability, and classification result.

Example response:

```json
{
    "prediction": 0,
    "fraud_probability": 0.02,
    "result": "Legitimate transaction"
}
```

For a fraudulent transaction:

```json
{
    "prediction": 1,
    "fraud_probability": 0.94,
    "result": "Fraudulent transaction"
}
```

## API Documentation

FastAPI automatically generates interactive API documentation using Swagger UI.

When running locally, the documentation is available at:

```text
http://127.0.0.1:8000/docs
```

After deployment:

```text
https://credit-card-fraud-detection-1-x62y.onrender.com/docs
```

## Running the Project Locally

### 1. Clone the Repository

```bash
git clone https://github.com/Vineetyadav07-42/credit-card-fraud-detection.git
cd credit_card_resume
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Start the FastAPI Application

```bash
uvicorn src.app:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

## Running with Docker

The application is containerized using Docker.

### Build the Docker Image

```bash
docker build -t credit-card-fraud-api .
```

### Run the Container

```bash
docker run -p 8000:8000 credit-card-fraud-api
```

The API will then be available at:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

## Deployment

The application is containerized using Docker and deployed as a web service on Render.

The deployment workflow is:

```text
GitHub Repository
       |
       v
     Render
       |
       v
 Docker Image Build
       |
       v
 Docker Container
       |
       v
 FastAPI Application
       |
       v
 Public REST API
```

### Live Application

API:

```text
https://credit-card-fraud-detection-1-x62y.onrender.com
```

Swagger documentation:

```text
https://credit-card-fraud-detection-1-x62y.onrender.com/docs
```

## Key Learning Outcomes

This project provided practical experience with:

* Binary classification
* Imbalanced classification problems
* Exploratory data analysis
* Model comparison
* Model evaluation
* Hyperparameter tuning
* Random Forest
* XGBoost
* Precision and recall trade-offs
* ROC-AUC and PR-AUC
* Model serialization using Joblib
* REST API development
* FastAPI
* Pydantic
* Docker
* Git and GitHub
* Cloud deployment




## Author

Vineet Yadav

GitHub: https://github.com/Vineetyadav07-42