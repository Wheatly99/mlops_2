import pandas as pd
import os

os.chdir('/home/vboxuser/Project/mlops_2/scripts/data-processing')
os.makedirs('../../data/stage2', exist_ok=True)

df = pd.read_csv('../../data/stage1/train.csv')
df.Age.fillna(df.Age.mean(), inplace=True)
df.to_csv('../../data/stage2/train.csv', index=False)