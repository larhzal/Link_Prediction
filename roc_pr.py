import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_all_roc_curves(y_true, y_pred_gcn, y_pred_gat, y_pred_sage):
    fpr_gcn, tpr_gcn, _ = roc_curve(y_true, y_pred_gcn)
    fpr_gat, tpr_gat, _ = roc_curve(y_true, y_pred_gat)
    fpr_sage, tpr_sage, _ = roc_curve(y_true, y_pred_sage)

    roc_auc_gcn = auc(fpr_gcn, tpr_gcn)
    roc_auc_gat = auc(fpr_gat, tpr_gat)
    roc_auc_sage = auc(fpr_sage, tpr_sage)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_gcn, tpr_gcn, color='green', lw=2, label=f'GCN (AUC = {roc_auc_gcn:.2f})')
    plt.plot(fpr_gat, tpr_gat, color='darkorange', lw=2, label=f'GAT (AUC = {roc_auc_gat:.2f})')
    plt.plot(fpr_sage, tpr_sage, color='blue', lw=2, label=f'GraphSAGE (AUC = {roc_auc_sage:.2f})')

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - GCN vs GAT vs GraphSAGE')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("roc_comparison.png", dpi=300)
    plt.show()
