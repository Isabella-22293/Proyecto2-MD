import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# 1. Carga
df = pd.read_csv("dataset_preprocesado.csv")
y = (df['review_score'] >= 4).astype(int)
X = df.drop(columns=['review_score']).select_dtypes(include=[np.number])

# 2. División
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Modelos y grids
modelos = {
    "Logistic": LogisticRegression(max_iter=1000),
    "KNN":      KNeighborsClassifier(algorithm='ball_tree'),
    "Tree":     DecisionTreeClassifier(),
    "RF":       RandomForestClassifier(),
    "GB":       GradientBoostingClassifier(),
    "SVC":      SVC(probability=True),
    "NB":       GaussianNB()
}
param_grids = {
    "Logistic": {'C':[0.01,0.1,1,10],'penalty':['l2'],'solver':['liblinear']},
    "Tree":     {'max_depth':[None,5,10],'min_samples_leaf':[1,5]},
    "RF":       {'n_estimators':[100,200],'max_depth':[None,10]},
    "GB":       {'n_estimators':[100,200],'learning_rate':[0.01,0.1]},
    "SVC":      {'C':[0.1,1,10],'kernel':['rbf','linear'],'gamma':['scale','auto']},
    "NB":       {}
}

# 4. Comparación sin CV vs CV
print("=== Clasificación sin CV ===")
for name, model in modelos.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{name}:")
    print(classification_report(y_test, y_pred, digits=3))

print("\n=== Clasificación con 5-Fold CV (ROC-AUC) ===")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, model in modelos.items():
    scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc', n_jobs=1)
    print(f"{name}: AUC_cv={scores.mean():.3f} ± {scores.std():.3f}")

# 5. Ajuste de hiperparámetros (Randomized para KNN)
best = {}
for name, model in modelos.items():
    print(f"\n==> Ajustando {name}...")
    if name == "KNN":
        rs = RandomizedSearchCV(
            estimator=model,
            param_distributions={'n_neighbors':range(3,8),'weights':['uniform','distance']},
            n_iter=5, cv=3, scoring='roc_auc', random_state=42, n_jobs=1
        )
        rs.fit(X_train, y_train)
        best[name] = rs.best_estimator_
        print("  Params:", rs.best_params_)
    else:
        gs = GridSearchCV(model, param_grids[name], cv=5, scoring='roc_auc', n_jobs=1)
        gs.fit(X_train, y_train)
        best[name] = gs.best_estimator_
        print("  Params:", gs.best_params_)

# 6. Evaluación final
print("\n=== Evaluación test tras ajuste ===")
for name, m in best.items():
    y_pred = m.predict(X_test)
    y_proba = m.predict_proba(X_test)[:,1] if hasattr(m, "predict_proba") else None
    print(f"\n-- {name} --")
    print(classification_report(y_test, y_pred, digits=4))
    if y_proba is not None:
        print("AUC:", roc_auc_score(y_test, y_proba))
    print("CM:\n", confusion_matrix(y_test, y_pred))
