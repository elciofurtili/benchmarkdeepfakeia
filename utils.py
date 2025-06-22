from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd

def generate_report(y_true, y_pred, filename="reports/metrics_report.csv"):
    report = {
        "Acuracia": accuracy_score(y_true, y_pred),
        "Precisao": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "AUC-ROC": roc_auc_score(y_true, y_pred)
    }
    df = pd.DataFrame([report])
    df.to_csv(filename, index=False)
    print(f"Relatório salvo em {filename}")