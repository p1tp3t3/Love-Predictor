import hashlib
import os

import joblib
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd


DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset.csv')
# Trained model saved with joblib, so the app doesn't retrain on every start
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'love_predictor.joblib')

# Features a person can realistically change: (how much to improve by, max value)
ACTIONABLE_FEATURES = {
    'social_activity': (2, 10),
    'confidence': (2, 10),
    'hobbies_count': (2, 10),
    'friends_count': (5, 100),
    'uses_dating_apps': (1, 1),
    'goes_out_per_month': (4, 30),
}

# Features explained in the insights. Age and sex are left out on purpose:
# they can't be changed and explaining them would feel judgy.
INSIGHT_FEATURES = [
    'past_relationships',
    'years_single',
    'social_activity',
    'confidence',
    'hobbies_count',
    'friends_count',
    'uses_dating_apps',
    'goes_out_per_month',
]

# Ignore answers that move the chance by less than 1 percentage point either way
MIN_INSIGHT_IMPACT = 0.01


class LovePredictor:
    # The LovePredictor class is responsible for handling the machine learning model and its operations.
    # It predicts whether the person will have a relationship based on the history of the situation of the person.
    def __init__(self, dataset_path=DATASET_PATH, model_path=MODEL_PATH):
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.model = None
        self.df = pd.read_csv(self.dataset_path)
        self.is_trained = False
        self.Y = 'will_have_relationship'
        self.X = [column for column in self.df.columns if column != self.Y]
        self.test_x = None
        self.test_y = None
        # Use the saved model if it still matches the dataset, otherwise train a new one
        if not self.load():
            self.train()

    def _split(self):
        # Same random_state every time, so a loaded model is tested on the same rows it was trained without
        return train_test_split(self.df[self.X], self.df[self.Y], test_size=0.2, random_state=42)

    def _dataset_hash(self):
        with open(self.dataset_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()

    def _finish_setup(self):
        _, self.test_x, _, self.test_y = self._split()
        # Everyone's predicted chance, used to compare a new person with the dataset
        self.dataset_probabilities = self.model.predict_proba(self.df[self.X])[:, 1]
        self.is_trained = True

    def train(self, save=True):
        # Scale the features first so LogisticRegression converges cleanly
        self.model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        train_x, _, train_y, _ = self._split()
        self.model.fit(train_x, train_y)
        self._finish_setup()
        if save:
            self.save()

    def save(self):
        joblib.dump({
            'model': self.model,
            'features': self.X,
            'dataset_hash': self._dataset_hash(),
            'sklearn_version': sklearn.__version__,
        }, self.model_path)

    def load(self):
        # Returns False when there is no usable saved model, so the caller can retrain
        if not os.path.exists(self.model_path):
            return False
        try:
            saved = joblib.load(self.model_path)
        except Exception:
            return False
        # Retrain if the data, the columns or the scikit-learn version changed since saving
        if (saved.get('dataset_hash') != self._dataset_hash()
                or saved.get('features') != self.X
                or saved.get('sklearn_version') != sklearn.__version__):
            return False
        self.model = saved['model']
        self._finish_setup()
        return True

    def evaluate(self):
        # Accuracy on the held-out test split
        return float(self.model.score(self.test_x, self.test_y))

    def _to_frame(self, data):
        missing = [column for column in self.X if column not in data]
        if missing:
            raise ValueError(f'Missing fields: {", ".join(missing)}')
        try:
            return pd.DataFrame([{column: float(data[column]) for column in self.X}])
        except (TypeError, ValueError):
            raise ValueError('All fields must be numeric')

    def append_data(self, data, save=False):
        # Only labeled rows can be used for training, so the label is required here
        if self.Y not in data:
            raise ValueError(f'Missing field: {self.Y}')
        df = self._to_frame(data)
        df[self.Y] = int(data[self.Y])
        # dont append duplicate
        if not self.df.merge(df, how='inner').empty:
            return False
        self.df = pd.concat([self.df, df], ignore_index=True)
        if save:
            self.df.to_csv(self.dataset_path, index=False)
        return True

    def predict(self, data):
        if not self.is_trained:
            raise RuntimeError('Model is not trained')
        features = self._to_frame(data)
        prediction = int(self.model.predict(features)[0])
        probability = float(self.model.predict_proba(features)[0][1])

        return {
            'prediction': prediction,
            'probability': probability,
            'insights': self.insights(features, probability),
        }

    def insights(self, features, probability):
        # Explains this person's prediction: how they compare with everyone in the dataset,
        # and which of their answers push their chance up (strengths) or down (challenges).
        factors = []
        for feature in INSIGHT_FEATURES:
            value = float(features.at[0, feature])
            average = float(self.df[feature].mean())
            # Swap this one answer for the average answer and see how the chance changes.
            # The difference is how much this answer helps or hurts, in percentage points.
            changed = features.copy()
            changed.at[0, feature] = average
            impact = probability - float(self.model.predict_proba(changed)[0][1])
            if abs(impact) < MIN_INSIGHT_IMPACT:
                continue
            factors.append({
                'feature': feature,
                'value': value,
                'average': average,
                'impact': impact,
            })

        strengths = sorted([f for f in factors if f['impact'] > 0], key=lambda f: f['impact'], reverse=True)
        challenges = sorted([f for f in factors if f['impact'] < 0], key=lambda f: f['impact'])

        return {
            'average_probability': float(self.dataset_probabilities.mean()),
            # Share of people in the dataset with a lower predicted chance
            'percentile': float((self.dataset_probabilities < probability).mean()),
            'strengths': strengths[:3],
            'challenges': challenges[:3],
        }

    def improvements(self, data, top_n=3):
        # What-if analysis: nudge each feature the person can change and measure how much
        # the predicted probability goes up. Returns the biggest gains first.
        if not self.is_trained:
            raise RuntimeError('Model is not trained')
        features = self._to_frame(data)
        base = float(self.model.predict_proba(features)[0][1])

        results = []
        for feature, (step, upper) in ACTIONABLE_FEATURES.items():
            current = float(features.at[0, feature])
            target = min(current + step, upper)
            if target <= current:
                continue
            changed = features.copy()
            changed.at[0, feature] = target
            gain = float(self.model.predict_proba(changed)[0][1]) - base
            if gain > 0:
                results.append({
                    'feature': feature,
                    'current': current,
                    'target': target,
                    'gain': gain,
                })

        results.sort(key=lambda item: item['gain'], reverse=True)
        return results[:top_n]


if __name__ == '__main__':
    # Retrain from dataset.csv and save the model: python -m model.love_predictor
    predictor = LovePredictor()
    predictor.train()
    print(f'Saved model to {predictor.model_path}')
    print(f'Test accuracy: {predictor.evaluate():.3f}')
