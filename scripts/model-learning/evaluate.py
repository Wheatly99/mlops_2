import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import yaml
import joblib
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import json
import os

os.chdir('/home/vboxuser/Project/mlops_2/scripts/model-learning')
os.makedirs('../../evaluate', exist_ok=True)

df = pd.read_csv('../../data/stage4/test.csv')
dt = joblib.load('../../models/dt.pkl')

X = df[['Pclass', 'Sex', 'Age']]
y = df.Survived

pred = dt.predict(X)

data = {"accuracy": accuracy_score(y, pred)}, {"precision": precision_score(y, pred)}, {"recall": recall_score(y, pred)}, {"f1": f1_score(y, pred)}

json_data = json.dumps(data)
with open('../../evaluate/score.json', 'w') as f:
    json.dump(json_data, f)