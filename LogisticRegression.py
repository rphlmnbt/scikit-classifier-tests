import pandas as pd
from sklearn.linear_model import LogisticRegression

class Logistic:
    def predict(self, inputArr):
        data = pd.read_csv("user.csv")
        X = data.drop(columns=['out'])
        y = data['out']

        model = LogisticRegression()
        model.fit(X, y)
        predictions = model.predict([inputArr])
        predictions_proba = model.predict_proba([inputArr])

        print('Logistic Regression Results:')
        print(predictions)
        print(predictions_proba)
