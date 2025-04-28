import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Carga
df = pd.read_csv("dataset_preprocesado.csv")

#Variables
y = df['review_score']
X = df.drop(columns=['review_score','order_id','customer_id','review_id'], errors='ignore')
X = X.select_dtypes(include=[np.number])

#Test
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)

#Modelos básicos
modelos = {
    "Linear": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "Tree": DecisionTreeRegressor(),
    "RF": RandomForestRegressor(),
    "GB": GradientBoostingRegressor(),
    "SVR": SVR()
}

#Evaluación inicial
def evalua(name, m):
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print(f"{name}: MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f}")

for n, mod in modelos.items():
    evalua(n, mod)

#Ajuste de parámetros
param_grids = {
    "Ridge": {'alpha':[0.1,1,10]},
    "Lasso": {'alpha':[0.01,0.1,1]},
    "Tree":  {'max_depth':[None,5,10]},
    "RF":    {'n_estimators':[100,200],'max_depth':[None,10]},
    "GB":    {'n_estimators':[100,200],'learning_rate':[0.01,0.1]},
    "SVR":   {'C':[0.1,1,10],'epsilon':[0.1,0.2]}
}
best = {}
for name in param_grids:
    gs = GridSearchCV(modelos[name], param_grids[name],
                      cv=5, scoring='neg_mean_squared_error')
    gs.fit(X_train, y_train)
    best[name] = gs.best_estimator_
    print(f"{name} mejores params:", gs.best_params_)

#Evaluación de modelos afinados
print("\nModelos afinados:")
for n, m in best.items():
    evalua(n, m)
