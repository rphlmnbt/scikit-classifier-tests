import pandas as pd
from input import *
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("user.csv")
X = data.drop(columns=['out'])
y = data['out']

model = DecisionTreeClassifier()
model.fit(X, y)
predictions = model.predict([inputArr])
predictions_proba = model.predict_proba([inputArr])

print(predictions)
print(predictions_proba)
