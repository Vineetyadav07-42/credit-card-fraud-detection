
# Credit Card Fraud Detection

An end-to-end machine learning project that detects fraudulent credit card transactions using traditional machine learning algorithms and PyTorch neural networks. The project includes model comparison, hyperparameter tuning, REST API development, Docker containerization, and AWS-based CI/CD deployment.


## Live Deployment

The Credit Card Fraud Detection API is deployed on AWS EC2.

- **Interactive API documentation (Swagger UI):** http://13.234.69.92:8001/docs
- **API base URL:** http://13.234.69.92:8001/

Use the Swagger UI to try the `POST /predict` endpoint and submit a transaction for prediction. The EC2 public IP can change if the instance is stopped and restarted without an Elastic IP, so update these links if the deployment address changes.



## Project Overview

Credit card fraud detection is an imbalanced binary classification problem where fraudulent transactions represent a very small proportion of all transactions. The goal is to identify fraudulent transactions while minimizing missed fraud and unnecessary false alarms.

This project compares several classification algorithms using stratified cross-validation, tunes the two best candidates using Precision-Recall AUC (PR-AUC), and serves the final Random Forest model through a FastAPI application.



## Key Features

- Class imbalance analysis
- Comparison of Logistic Regression, Decision Tree, Random Forest, XGBoost, and a PyTorch neural network
- Leak-free stratified 5-fold cross-validation (scaling fitted inside each fold, identical folds for all models)
- Randomized hyperparameter search for Random Forest and the neural network (4 configurations each)
- PyTorch neural network with mini-batch training
- Evaluation using precision, recall, F1-score, ROC-AUC, and PR-AUC
- REST API built with FastAPI
- Docker containerization
- AWS ECR and EC2 deployment
- GitHub Actions CI/CD workflow for building, publishing, and deploying the image



## Tech Stack

| Category             | Technologies          |
| -------------------- | --------------------- |
| Language             | Python                |
| Data processing      | Pandas, NumPy         |
| Machine learning     | Scikit-learn, XGBoost |
| Deep learning        | PyTorch               |
| Model evaluation     | Scikit-learn metrics  |
| API development      | FastAPI, Pydantic     |
| Model serialization  | Joblib                |
| Containerization     | Docker                |
| Cloud infrastructure | AWS ECR, AWS EC2      |
| CI/CD                | GitHub Actions        |
| Version control      | Git, GitHub           |




## Dataset

[Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud): 284,807 transactions by European cardholders, of which 492 (about 0.17%) are fraudulent.

- `Time`: Seconds elapsed between the transaction and the first transaction in the dataset.
- `Amount`: Transaction amount.
- `V1`–`V28`: Anonymized numerical features (PCA components).
- `Class`: Target variable (`0` = legitimate, `1` = fraudulent).

The dataset is highly imbalanced, so accuracy alone is insufficient for evaluating model performance.

The dataset is not included in this repository. Download it and place it at `data/creditcard.csv`.


## Project Workflow

1. **Data preparation:** Load the dataset, inspect the class distribution, and separate features from the target.
2. **Train-test split:** Create an 80% training set and a 20% holdout test set using stratified sampling. The test set is not used until the final evaluation.
3. **Model selection:** Compare multiple classifiers using identical stratified 5-fold cross-validation splits on the training set.
4. **Preprocessing:** Standardize `Time` and `Amount`, fitting the scaler on the training portion of each fold only (a Pipeline for scikit-learn models, manual per-fold scaling for the neural network).
5. **Hyperparameter tuning:** Tune the two finalists with the same budget: `RandomizedSearchCV` for Random Forest and a custom randomized search for the neural network, both scored on PR-AUC.
6. **Model evaluation:** Use PR-AUC as the primary selection metric and review precision, recall, and F1-score alongside it.
7. **Final evaluation and model choice:** Evaluate the tuned models once on the holdout test set. Random Forest and the neural network tied, so Random Forest was chosen for simplicity.
8. **Final training:** Refit the chosen Random Forest configuration on the full dataset and save it with Joblib. The test metrics reported below come from the same configuration trained on the training split only.
9. **API development:** Load the serialized model in FastAPI and expose prediction endpoints.
10. **Deployment:** Package the application and model into a Docker image, publish it to Amazon ECR, and run it on EC2 through GitHub Actions.


## Models Implemented

### Traditional Machine Learning

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- XGBoost Classifier

### Neural Network

A feed-forward binary classification network in PyTorch (ReLU activations, optional dropout, Adam optimizer, `BCEWithLogitsLoss`). The baseline uses one hidden layer with 10 units. The tuning search varies hidden-layer sizes, dropout, learning rate, weight decay, batch size, epochs, and positive-class weighting.

### Hyperparameter Tuning

Randomized search with cross-validation on the training set only. Each finalist was given the same budget of 4 configurations x 5 folds. PR-AUC is the tuning metric because the dataset contains very few fraudulent transactions.

## Results

### Baseline CV (5-fold stratified, training set)

| Model                                     | PR-AUC (mean ± std) |
| ----------------------------------------- | ------------------- |
| Random Forest                             | 0.840 ± 0.032       |
| Neural network (1 hidden layer, 10 units) | 0.834 ± 0.037       |
| Logistic Regression                       | 0.762 ± 0.042       |
| XGBoost (defaults)                        | 0.749 ± 0.029       |
| Decision Tree                             | 0.598 ± 0.022       |

Random Forest and the neural network tied (the gap is smaller than one standard deviation), so both were tuned.


### Held-out test set (56,962 transactions, 98 frauds, threshold 0.5)

| Model (tuned)  | PR-AUC | Precision | Recall | F1    | TP / FP / FN |
| -------------- | ------ | --------- | ------ | ----- | ------------ |
| Random Forest  | 0.853  | 0.892     | 0.755  | 0.818 | 74 / 9 / 24  |
| Neural network | 0.848  | 0.884     | 0.776  | 0.826 | 76 / 10 / 22 |

The models tied again, so **Random Forest was selected for simplicity**: no feature scaling required, faster training and inference, and a lighter container than a PyTorch model.

Final Random Forest parameters:

```python
n_estimators=200, min_samples_leaf=2, max_samples=0.5,
max_features=0.3, max_depth=20, class_weight="balanced_subsample"
```

## Evaluation Metrics

| Metric    | Purpose                                                                                          |
| --------- | ------------------------------------------------------------------------------------------------ |
| Precision | Measures how many predicted fraud cases are actually fraudulent.                                 |
| Recall    | Measures how many actual fraud cases are detected.                                               |
| F1-score  | Balances precision and recall.                                                                   |
| ROC-AUC   | Measures the model's ability to rank positive cases above negative cases.                        |
| PR-AUC    | Evaluates precision-recall performance, particularly useful for imbalanced classification.       |
| Accuracy  | Not used for model selection: predicting "legitimate" for everything already scores about 99.8%. |

**Primary metric: PR-AUC.** Precision and recall matter as well, because missed fraud and false alarms have different operational costs.




## Repository Structure

```text
credit-card-fraud-detection/
├── data/
│   └── creditcard.csv
├── notebooks/
│   ├── Choosing_model.ipynb
│   ├── fine_tune_randomforest.ipynb
│   ├── neural_network.ipynb
│   └── fine_tune_nn.ipynb
├── src/
│   ├── app.py
│   └── training.py
├── .github/
│   └── workflows/
│       └── deploy.yml
├── model.joblib
├── Dockerfile
├── requirements.txt
├── .gitignore
├── .dockerignore
└── README.md
```


## Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/Vineetyadav07-42/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the API

From the repository root:

```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

Open the interactive API documentation at `http://localhost:8000/docs`.

The trained model (`model.joblib`) is included in the repository, so the dataset is only needed to rerun the notebooks.


## API Usage

| Method | Endpoint   | Description                          |
| ------ | ---------- | ------------------------------------ |
| GET    | `/`        | Status message                       |
| POST   | `/predict` | Fraud prediction for one transaction |


### Prediction endpoint

**POST** `/predict`

The request body must contain all 30 features: `Time`, `V1` through `V28`, and `Amount`.

Example request (the first transaction in the dataset, which is legitimate):

```json
{
  "Time": 0.0,
  "V1": -1.3598, "V2": -0.0728, "V3": 2.5363, "V4": 1.3782,
  "V5": -0.3383, "V6": 0.4624, "V7": 0.2396, "V8": 0.0987,
  "V9": 0.3638, "V10": 0.0908, "V11": -0.5516, "V12": -0.6178,
  "V13": -0.9914, "V14": -0.3112, "V15": 1.4682, "V16": -0.4704,
  "V17": 0.2080, "V18": 0.0258, "V19": 0.4040, "V20": 0.2514,
  "V21": -0.0183, "V22": 0.2778, "V23": -0.1105, "V24": 0.0669,
  "V25": 0.1285, "V26": -0.1891, "V27": 0.1336, "V28": -0.0211,
  "Amount": 149.62
}
```

Response fields:

| Field               | Type   | Description                                              |
| ------------------- | ------ | -------------------------------------------------------- |
| `prediction`        | int    | Predicted class: `1` = fraud, `0` = legitimate           |
| `fraud_probability` | float  | Model probability that the transaction is fraudulent     |
| `result`            | string | `"Fraudulent transaction"` or `"Legitimate transaction"` |

## Docker

Build the Docker image:

```bash
docker build -t credit-card-fraud .
```

Run the container:

```bash
docker run -d \
  --name credit-card-fraud-api \
  -p 8001:8000 \
  credit-card-fraud
```

Access the API at `http://localhost:8001/docs`.



## AWS Deployment and CI/CD

When code is pushed to the `main` branch, GitHub Actions:

1. Checks out the repository.
2. Sets up Python and installs dependencies (the test step is a placeholder for now).
3. Configures AWS credentials and logs in to Amazon ECR.
4. Builds the Docker image and pushes it to ECR with the `latest` tag.
5. Connects to the EC2 instance over SSH, pulls the new image, stops and removes the old container, and starts the new one on port 8001.

### Required GitHub Secrets

Configure these in **Settings → Secrets and variables → Actions**:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`
- `ECR_REPOSITORY`
- `EC2_HOST`
- `EC2_USERNAME`
- `EC2_SSH_KEY`

The EC2 instance needs Docker, the AWS CLI, and an IAM role that allows pulling from ECR. Its security group must allow inbound traffic on port 8001.

## Limitations

- **Fixed decision threshold (0.5):** Recall is about 0.76, so roughly one in four frauds in the test set was missed. The threshold could be tuned on out-of-fold predictions to trade precision for recall.
- **Light tuning:** Only 4 configurations per model were searched, so better settings may exist.
- **No automated tests yet:** The CI test step is a placeholder, and the deployment does not verify that the new container is healthy after the restart.
- **Small test set:** The test set contains 98 frauds, so differences of a few frauds between models are within noise.
- **Deployed model has no separate held-out score:** It uses the same hyperparameters as the evaluated model but is retrained on all data.
- **Dataset:** The features are PCA-anonymized and come from a single, older dataset. Real-world use would require monitoring and retraining.

## Future Improvements

- Add automated unit and API tests with `pytest` and run them in CI.
- Add a `/health` endpoint and a post-deploy health check.
- Tag Docker images with the commit SHA to allow rollback to a previous release.
- Tune the classification threshold on validation data.
- Add model and data validation checks to the CI/CD workflow.
- Monitor prediction distributions and model performance over time.
- Add structured logging and richer API error handling.
- Track experiments and model versions using MLflow.
- Evaluate inference latency and resource usage under load.

## Disclaimer

This project is intended for educational and portfolio purposes. It is not a production-certified financial fraud detection system. Predictions should not be used as the sole basis for real financial decisions.
