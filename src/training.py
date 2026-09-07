import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH=os.path.join(BASE_DIR,'data','creditcard.csv')
MODEL_PATH=os.path.join(BASE_DIR,'model.pkl')

data=pd.read_csv(DATA_PATH)
X=data.drop(columns=['Class'])
y=data['Class']

model=RandomForestClassifier(n_estimators=200,
                             max_features='sqrt',
                             random_state=42,
                             max_depth=None)

model.fit(X,y)

joblib.dump(model,MODEL_PATH)

print('Model successfully trained and saved!')