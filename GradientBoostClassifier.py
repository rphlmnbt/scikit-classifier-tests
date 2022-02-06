import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

class GradientBoost:
    def predict(self, inputArr):
        data = pd.read_csv("user.csv")
        X = data.drop(columns=['out'])
        y = data['out']

        model = GradientBoostingClassifier()
        model.fit(X, y)
        predictions = model.predict([inputArr])
        predictions_proba = model.predict_proba([inputArr])

        print('Gradient Boost Classifier Results:')
        print(predictions)
        print(predictions_proba)
