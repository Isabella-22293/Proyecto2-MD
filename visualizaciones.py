import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (
    r2_score, roc_curve, auc,
    confusion_matrix
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPClassifier

# 1. Carga y split
df = pd.read_csv("dataset_preprocesado.csv")

# Para regresión
y_reg = df['review_score']
X_reg = df.drop(columns=['review_score']).select_dtypes(include=[np.number])
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Para clasificación
y_clf = (df['review_score'] >= 4).astype(int)
X_clf = X_reg.copy()
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(X_clf, y_clf, test_size=0.2,
                                              stratify=y_clf, random_state=42)

# 2. Entrenar modelos seleccionados
# Regresión
rf_reg = RandomForestRegressor(random_state=42).fit(Xr_tr, yr_tr)

# Clasificación
rf_clf = RandomForestClassifier(random_state=42).fit(Xc_tr, yc_tr)

# Red neuronal simple
mlp100 = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
mlp100.fit(Xc_tr, yc_tr)

# Modelos para sesgo-varianza
gb = GradientBoostingRegressor(random_state=42)
svr = SVR()
mlp100_100 = MLPClassifier(hidden_layer_sizes=(100,100), max_iter=500, random_state=42)

# 3. Barras comparativas R² vs AUC
r2 = r2_score(yr_te, rf_reg.predict(Xr_te))
# Reutilizamos rf_clf y mlp100 para AUC
proba_rf = rf_clf.predict_proba(Xc_te)[:,1]
fpr, tpr, _ = roc_curve(yc_te, proba_rf)
auc_rf = auc(fpr, tpr)

proba_mlp = mlp100.predict_proba(Xc_te)[:,1]
fpr2, tpr2, _ = roc_curve(yc_te, proba_mlp)
auc_mlp = auc(fpr2, tpr2)

models_r = ['RandomForestReg']
scores_r  = [r2]
models_c = ['RF Clf','MLP(100)']
scores_c  = [auc_rf, auc_mlp]

x1 = np.arange(len(models_r))
x2 = np.arange(len(models_c)) + len(models_r) + 1

plt.figure(figsize=(8,4))
plt.bar(x1, scores_r, label='R² Regresión')
plt.bar(x2, scores_c, label='AUC Clasificación/NN')
plt.xticks(np.concatenate([x1,x2]), models_r + models_c, rotation=45, ha='right')
plt.ylabel('Puntuación')
plt.title('Comparativa R² vs ROC‑AUC')
plt.legend()
plt.tight_layout()
plt.show()

# 4. Curvas ROC superpuestas
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f'RF (AUC={auc_rf:.2f})')
plt.plot(fpr2, tpr2, label=f'MLP(100) (AUC={auc_mlp:.2f})')
plt.plot([0,1],[0,1],'k--', label='No Skill')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curvas ROC Comparadas')
plt.legend(loc='lower right')
plt.show()

# 5. Matriz de confusión de RandomForest
cm = confusion_matrix(yc_te, rf_clf.predict(Xc_te))
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicción')
plt.ylabel('Verdadero')
plt.title('Matriz de Confusión: RandomForest')
plt.tight_layout()
plt.show()

# 6. Sesgo‑Varianza: boxplot de AUC CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {
    'GB':    cross_val_score(gb,    X_reg, y_reg, cv=KFold(5,shuffle=True,random_state=42), scoring='r2'),
    'SVR':   cross_val_score(svr,   X_reg, y_reg, cv=KFold(5,shuffle=True,random_state=42), scoring='r2'),
}
# Para clasificación/NN
cv_results['MLP(100,100)'] = cross_val_score(
    mlp100_100, X_clf, y_clf, cv=skf, scoring='roc_auc')

plt.figure(figsize=(6,4))
sns.boxplot(data=list(cv_results.values()))
plt.xticks(range(len(cv_results)), list(cv_results.keys()), rotation=45, ha='right')
plt.ylabel('Score CV')
plt.title('Sesgo‑Varianza: Distribución de Scores')
plt.tight_layout()
plt.show()
