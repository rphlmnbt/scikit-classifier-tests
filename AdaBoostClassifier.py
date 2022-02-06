import pandas as pd
from sklearn.ensemble import AdaBoostClassifier

class AdaBoost:
    def predict(self, inputArr):
        data = pd.read_csv("user.csv")
        X = data.drop(columns=['out'])
        y = data['out']

        model = AdaBoostClassifier()
        model.fit(X, y)
        predictions = model.predict([inputArr])
        predictions_proba = model.predict_proba([inputArr])

        print('Ada Boost Classifier Results:')
        print(predictions)
        print(predictions_proba)
