import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# 1. Carga y preparación
df = pd.read_csv("dataset_preprocesado.csv")
y  = (df['review_score'] >= 4).astype(int)
X  = df.drop('review_score', axis=1).select_dtypes(include=[np.number])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Modelos y grid
modelos = {
    "Logistic": LogisticRegression(),
    "KNN":      KNeighborsClassifier(),
    "Tree":     DecisionTreeClassifier(),
    "RF":       RandomForestClassifier(),
    "GB":       GradientBoostingClassifier(),
    "SVC":      SVC(probability=True),
    "NB":       GaussianNB()
}
# param_grids como en sección 1.2

# 3. GridSearch + guardar mejores modelos
best_models = {}
for name, model in modelos.items():
    gs = GridSearchCV(model, param_grids[name], cv=5,
                      scoring='roc_auc', n_jobs=-1)
    gs.fit(X_train, y_train)
    best_models[name] = gs.best_estimator_
    print(f"{name} mejores params: {gs.best_params_}")