import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    mean_squared_error,
    r2_score
)

def plot_confusion(name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(f"Confusion Matrix: {name}")
    plt.show()

def plot_roc(name, y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")
    plt.plot([0,1],[0,1],'--')
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

def regression_metrics(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    print(f"{name}: MSE={mse:.3f}, R2={r2:.3f}")
