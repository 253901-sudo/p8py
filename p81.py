import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.stats import loguniform

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=5000)

cv = RepeatedStratifiedKFold(
    n_splits=5, n_repeats=2, random_state=42
)

param_grid = {
    'solver': ['liblinear'],
    'penalty': ['l1', 'l2'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100]
}

grid_search = GridSearchCV(
    model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=cv,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Classification - Grid Search")
print("Best Parameters:", grid_search.best_params_)
print("Best CV Score:", grid_search.best_score_)
print("Test Accuracy:", accuracy_score(y_test, grid_search.predict(X_test)))

param_dist = [
    {
        'solver': ['liblinear'],
        'penalty': ['l1', 'l2'],
        'C': loguniform(1e-5, 100)
    },
    {
        'solver': ['lbfgs'],
        'penalty': ['l2'],
        'C': loguniform(1e-5, 100)
    }
]

random_search = RandomizedSearchCV(
    model,
    param_distributions=param_dist,
    n_iter=20,
    scoring='accuracy',
    cv=cv,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

print("\nClassification - Random Search")
print("Best Parameters:", random_search.best_params_)
print("Best CV Score:", random_search.best_score_)
print("Test Accuracy:", accuracy_score(y_test, random_search.predict(X_test)))