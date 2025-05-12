import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

# 1. Carga y split
df = pd.read_csv("dataset_preprocesado.csv")
y = (df['review_score'] >= 4).astype(int)
X = df.drop(columns=['review_score']).select_dtypes(include=[np.number])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Escalado
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# 3. Arquitecturas a probar
architectures = {
    "100":      (100,),
    "100-50":   (100,50),
    "100-100":  (100,100),
}

print("=== Redes Neuronales sin CV ===")
for name, arch in architectures.items():
    mlp = MLPClassifier(hidden_layer_sizes=arch, max_iter=500, random_state=42)
    mlp.fit(X_train_s, y_train)
    y_pred = mlp.predict(X_test_s)
    y_proba = mlp.predict_proba(X_test_s)[:,1]
    print(f"{name}:")
    print(classification_report(y_test, y_pred, digits=3))
    print("AUC:", roc_auc_score(y_test, y_proba))

print("\n=== Redes Neuronales con CV (AUC) ===")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, arch in architectures.items():
    mlp = MLPClassifier(hidden_layer_sizes=arch, max_iter=500, random_state=42)
    scores = cross_val_score(mlp, scaler.fit_transform(X), y, cv=skf, scoring='roc_auc', n_jobs=1)
    print(f"{name}: AUC_cv={scores.mean():.3f} ± {scores.std():.3f}")
