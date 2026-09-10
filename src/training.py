from pathlib import Path
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

BASE_DIR=Path(__file__).resolve().parent.parent
DATA_PATH=BASE_DIR/'data'/'creditcard.csv'
MODEL_PATH=BASE_DIR/'model.pkl'

data=pd.read_csv(DATA_PATH)


X=data.drop(columns=['Class'])
y=data['Class']

model=RandomForestClassifier(n_estimators=200,
                            random_state=42,
                            max_depth=None)
                             

model.fit(X,y)

joblib.dump(model,MODEL_PATH)

print('Model successfully trained and saved!')