import numpy as np
import pandas as pd

# Generates a synthetic dataset for the LovePredictor.
# All columns are numeric so they can be fed straight into LogisticRegression.
# sex: 0 = male, 1 = female

N_ROWS = 1000
SEED = 42

rng = np.random.default_rng(SEED)

age = rng.integers(18, 61, N_ROWS)
sex = rng.choice([0, 1], N_ROWS, p=[0.5, 0.5])
past_relationships = np.clip(rng.poisson((age - 16) / 6), 0, 15)
years_single = np.clip(np.round(rng.exponential(2.5, N_ROWS), 1), 0, age - 18)
social_activity = rng.integers(1, 11, N_ROWS)
confidence = rng.integers(1, 11, N_ROWS)
hobbies_count = rng.integers(0, 11, N_ROWS)
friends_count = np.clip(rng.poisson(8 + social_activity * 1.5), 0, None)
uses_dating_apps = rng.binomial(1, 0.45, N_ROWS)
goes_out_per_month = np.clip(rng.poisson(social_activity * 0.8), 0, 30)

# Hidden logistic relationship that determines the label
logit = (
    -4.0
    + 0.25 * social_activity
    + 0.30 * confidence
    + 0.08 * hobbies_count
    + 0.03 * friends_count
    + 0.70 * uses_dating_apps
    + 0.06 * goes_out_per_month
    + 0.10 * np.minimum(past_relationships, 5)
    - 0.20 * years_single
    - 0.02 * np.abs(age - 30)
    + rng.normal(0, 0.8, N_ROWS)
)
probability = 1 / (1 + np.exp(-logit))
will_have_relationship = rng.binomial(1, probability)

df = pd.DataFrame({
    'age': age,
    'sex': sex,
    'past_relationships': past_relationships,
    'years_single': years_single,
    'social_activity': social_activity,
    'confidence': confidence,
    'hobbies_count': hobbies_count,
    'friends_count': friends_count,
    'uses_dating_apps': uses_dating_apps,
    'goes_out_per_month': goes_out_per_month,
    'will_have_relationship': will_have_relationship,
})

if __name__ == '__main__':
    df.to_csv('dataset.csv', index=False)
    print(f'Wrote {len(df)} rows to dataset.csv')
    print(df['will_have_relationship'].value_counts(normalize=True))
