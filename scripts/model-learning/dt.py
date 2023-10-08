import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import yaml
import joblib
import os

os.chdir('/home/vboxuser/Project/mlops_2/scripts/model-learning')
os.makedirs('../../models', exist_ok=True)

df = pd.read_csv('../../data/stage4/train.csv')
params = yaml.safe_load(open("../../params.yaml"))["train"]

seed = params["seed"]
max_depth = params["max_depth"]

X = df[['Survived', 'Pclass', 'Sex', 'Age']]
y = df.Survived

dt = DecisionTreeClassifier(max_depth=max_depth, random_state=seed)
dt.fit(X, y)

joblib.dump(dt, "../../models/dt.pkl")