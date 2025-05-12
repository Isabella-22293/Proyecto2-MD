import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Carga
df = pd.read_csv("dataset_preprocesado.csv")
y = df['review_score']
X = df.drop(columns=['review_score'], errors='ignore').select_dtypes(include=[np.number])

# 2. División
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Modelos
modelos = {
    "Linear": LinearRegression(),
    "Ridge":  Ridge(),
    "Lasso":  Lasso(),
    "Tree":   DecisionTreeRegressor(),
    "RF":     RandomForestRegressor(),
    "GB":     GradientBoostingRegressor(),
    "SVR":    SVR()
}
param_grids = {
    "Ridge": {'alpha':[0.1,1,10]},
    "Lasso": {'alpha':[0.01,0.1,1]},
    "Tree":  {'max_depth':[None,5,10]},
    "RF":    {'n_estimators':[100,200],'max_depth':[None,10]},
    "GB":    {'n_estimators':[100,200],'learning_rate':[0.01,0.1]},
    "SVR":   {'C':[0.1,1,10],'epsilon':[0.1,0.2]}
}

# 4. Evaluación sin CV
print("=== Sin validación cruzada ===")
for name, model in modelos.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{name}: MAE={mean_absolute_error(y_test,y_pred):.3f}, RMSE={np.sqrt(mean_squared_error(y_test,y_pred)):.3f}, R2={r2_score(y_test,y_pred):.3f}")

# 5. Evaluación con CV
print("\n=== Con 5-Fold CV (R2 promedio) ===")
cv = KFold(n_splits=5, shuffle=True, random_state=42)
for name, model in modelos.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring='r2', n_jobs=1)
    print(f"{name}: R2_cv={scores.mean():.3f} ± {scores.std():.3f}")

# 6. Ajuste de hiperparámetros
best = {}
for name in param_grids:
    gs = GridSearchCV(modelos[name], param_grids[name], cv=5, scoring='r2', n_jobs=1)
    gs.fit(X_train, y_train)
    best[name] = gs.best_estimator_
    print(f"{name} mejores params: {gs.best_params_}")

# 7. Evaluación de modelos afinados
print("\n=== Modelos afinados (sin CV) ===")
for name, model in best.items():
    y_pred = model.predict(X_test)
    print(f"{name}: R2={r2_score(y_test,y_pred):.3f}")
