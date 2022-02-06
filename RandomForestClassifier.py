import pandas as pd
from sklearn.ensemble import RandomForestClassifier

class RandomForest:
    def predict(self, inputArr):
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
