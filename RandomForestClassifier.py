import pandas as pd
from input import *
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("user.csv")
X = data.drop(columns=['out'])
y = data['out']

model = RandomForestClassifier()
model.fit(X, y)
predictions = model.predict([inputArr])
predictions_proba = model.predict_proba([inputArr])

print('Random Forest Classifier Results:')
print(predictions)
print(predictions_proba)
