
import pandas as pd
from google.colab import files

uploaded = files.upload()

df = pd.read_csv("heart_v2.csv")
print(df)

features = ['age','sex','BP','cholestrol']

X = df[features]
y = df['heart disease']

print(X)
print(y)

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)

tree.plot_tree(dtree, feature_names=features)

