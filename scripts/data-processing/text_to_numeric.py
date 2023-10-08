import pandas as pd
import os

os.chdir('/home/vboxuser/Project/mlops_2/scripts/data-processing')
os.makedirs('../../data/stage3', exist_ok=True)

df = pd.read_csv('../../data/stage2/train.csv')

df.loc[df.Sex == 'male', 'Sex'] = 1
df.loc[df.Sex == 'female', 'Sex'] = 0
df.Sex = df.Sex.astype(int)

df.to_csv('../../data/stage3/train.csv', index=False)