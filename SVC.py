import pandas as pd
from input import *
from sklearn.svm import SVC

data = pd.read_csv("user.csv")
X = data.drop(columns=['out'])
y = data['out']

model = SVC(probability=True)
model.fit(X, y)
predictions = model.predict([inputArr])
predictions_proba = model.predict_proba([inputArr])

print('SVC Results:')
print(predictions)
print(predictions_proba)
