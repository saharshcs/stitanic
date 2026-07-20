"""
Spaceship Titanic - simplified single-model solution.

Predicts whether each passenger was transported to another dimension.
This is a BINARY CLASSIFICATION problem (not regression): the target
`Transported` is True/False.

Model: CatBoost, a gradient-boosted decision tree classifier.
Validation: 5-fold stratified cross-validation.

The full pipeline (model_v6.py) stacks five models and scores ~0.820.
This single-model version scores slightly lower but is short enough to
read top to bottom in a few minutes.
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------------
# Settings
# ---------------------------------------------------------------
N_FOLDS = 5
SEED = 42
SPEND_COLS = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
CAT_COLS = ['HomePlanet', 'Destination', 'Deck', 'Side']

# ---------------------------------------------------------------
# 1. Load the data
# ---------------------------------------------------------------
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
n_train = len(train)

# Train and test are processed together so that group-level statistics
# (like group size) are correct: a travel group can have some members in
# train.csv and others in test.csv.
both = pd.concat([train, test], ignore_index=True)


# ---------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------
def engineer(df):
    df = df.copy()

    # PassengerId looks like "0013_01": group 0013, member 01.
    # Passengers in the same group usually share a fate, so group
    # information is highly predictive.
    df['Group'] = df['PassengerId'].str.split('_').str[0]
    df['GroupSize'] = df.groupby('Group')['Group'].transform('count')
    df['IsAlone'] = (df['GroupSize'] == 1).astype(int)

    # Cabin looks like "B/123/S": deck B, room 123, starboard side.
    df[['Deck', 'CabinNum', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df['CabinNum'] = pd.to_numeric(df['CabinNum'], errors='coerce')

    # A passenger in cryo-sleep cannot spend money, so spending tells us
    # what a missing CryoSleep value must have been.
    spent = df[SPEND_COLS].sum(axis=1, skipna=True)
    df.loc[df['CryoSleep'].isna() & (spent > 0), 'CryoSleep'] = False
    df.loc[df['CryoSleep'].isna() & (spent == 0), 'CryoSleep'] = True

    # Fill missing values.
    df[SPEND_COLS] = df[SPEND_COLS].fillna(0)
    df['CryoSleep'] = df['CryoSleep'].fillna(False).astype(int)
    df['VIP'] = df['VIP'].fillna(False).astype(int)
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['CabinNum'] = df['CabinNum'].fillna(df['CabinNum'].median())
    for col in CAT_COLS:
        df[col] = df[col].fillna('Unknown')

    # Spending features. Total spend separates the two classes strongly:
    # passengers who spent nothing were transported far more often.
    df['TotalSpend'] = df[SPEND_COLS].sum(axis=1)
    df['NoSpend'] = (df['TotalSpend'] == 0).astype(int)
    df['ServicesUsed'] = (df[SPEND_COLS] > 0).sum(axis=1)

    # Spending is heavily skewed (most people spend 0, a few spend
    # thousands), so log-scale it to compress the large values.
    df['LogTotalSpend'] = np.log1p(df['TotalSpend'])
    for col in SPEND_COLS:
        df['Log' + col] = np.log1p(df[col])

    df['IsChild'] = (df['Age'] < 13).astype(int)
    return df


both = engineer(both)

FEATURES = (
    CAT_COLS
    + ['CryoSleep', 'VIP', 'Age', 'IsChild', 'CabinNum', 'GroupSize', 'IsAlone']
    + ['TotalSpend', 'LogTotalSpend', 'NoSpend', 'ServicesUsed']
    + SPEND_COLS
    + ['Log' + c for c in SPEND_COLS]
)

X = both[FEATURES].iloc[:n_train]
X_test = both[FEATURES].iloc[n_train:]
y = train['Transported'].astype(int)

print(f'Training rows: {len(X)}   Test rows: {len(X_test)}   Features: {len(FEATURES)}')

# ---------------------------------------------------------------
# 3. Cross-validated training
# ---------------------------------------------------------------
# Stratified K-fold splits the data into 5 parts with the same class
# balance in each. We train on 4 parts and validate on the 5th, rotating
# through all of them. Every training row therefore gets one prediction
# from a model that never saw it ("out-of-fold"), which is an honest
# estimate of accuracy on unseen data.
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_preds = np.zeros(len(X))       # out-of-fold predictions for scoring
test_preds = np.zeros(len(X_test))  # averaged test predictions

for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
    X_tr, X_va = X.iloc[train_idx], X.iloc[valid_idx]
    y_tr, y_va = y.iloc[train_idx], y.iloc[valid_idx]

    model = CatBoostClassifier(
        iterations=2000,        # maximum number of trees
        learning_rate=0.05,     # how much each tree contributes
        depth=6,                # maximum tree depth
        loss_function='Logloss',
        cat_features=CAT_COLS,  # CatBoost handles these natively
        random_seed=SEED,
        verbose=False,
        allow_writing_files=False,
    )

    # early_stopping_rounds halts training once validation performance
    # stops improving, which prevents overfitting.
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=100)

    oof_preds[valid_idx] = model.predict_proba(X_va)[:, 1]
    test_preds += model.predict_proba(X_test)[:, 1] / N_FOLDS

    fold_acc = accuracy_score(y_va, oof_preds[valid_idx] > 0.5)
    print(f'Fold {fold + 1}: accuracy = {fold_acc:.4f}')

cv_accuracy = accuracy_score(y, oof_preds > 0.5)
print(f'\nCross-validated accuracy: {cv_accuracy:.4f}')

# ---------------------------------------------------------------
# 4. Which features mattered?
# ---------------------------------------------------------------
importances = pd.Series(model.get_feature_importance(), index=FEATURES)
print('\nTop 10 features:')
print(importances.sort_values(ascending=False).head(10).round(2).to_string())

# ---------------------------------------------------------------
# 5. Write the submission
# ---------------------------------------------------------------
# Probabilities above 0.5 become True.
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Transported': test_preds > 0.5,
})
submission.to_csv('final_submission.csv', index=False)
print(f'\nWrote final_submission.csv ({len(submission)} rows)')
