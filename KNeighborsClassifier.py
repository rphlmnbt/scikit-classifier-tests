import pandas as pd
from input import *
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("user.csv")
X = data.drop(columns=['out'])
y = data['out']

model = KNeighborsClassifier()
model.fit(X, y)
predictions = model.predict([inputArr])
predictions_proba = model.predict_proba([inputArr])

print('KNeighors Classifier Results:')
print(predictions)
print(predictions_proba)
