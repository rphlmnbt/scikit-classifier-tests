import pandas as pd
from sklearn.tree import DecisionTreeClassifier

class DecisionTree:
    def predict(self, inputArr):
        data = pd.read_csv("user.csv")
        X = data.drop(columns=['out'])
        y = data['out']

        model = DecisionTreeClassifier()
        model.fit(X, y)
        predictions = model.predict([inputArr])
        predictions_proba = model.predict_proba([inputArr])

        print('Decision Tree Results:')
        print(predictions)
        print(predictions_proba)
