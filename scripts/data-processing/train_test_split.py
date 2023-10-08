import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import os

os.chdir('/home/vboxuser/Project/mlops_2/scripts/data-processing')
os.makedirs('../../data/stage4', exist_ok=True)

df = pd.read_csv('../../data/stage3/train.csv')
params = yaml.safe_load(open("../../params.yaml"))["split"]

p_split = params["split_ratio"]

X = df[['Survived', 'Pclass', 'Sex', 'Age']]
y = df.Survived

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = p_split, stratify=y)

pd.concat([X_train, y_train], axis=1).to_csv('../../data/stage4/train.csv', index=False)
pd.concat([X_test, y_test], axis=1).to_csv('../../data/stage4/test.csv', index=False)