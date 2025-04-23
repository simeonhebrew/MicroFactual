import os
import pandas as pd

from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

from microbiome_ml.utils import get_logger
app_logger = get_logger(__name__)

def plot_roc(y_true, y_probs, save_path=None, show=False):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = roc_auc_score(y_true, y_probs)

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend()
    if save_path:
        save_path = os.path.join(save_path, 'roc_curve.png')
        plt.savefig(save_path)
        app_logger.info(f"ROC curve saved to {save_path}")
    if show:
        plt.show()
        
def save_roc_curve(output_dir: str, labels: pd.Series, probs: pd.Series, logger=get_logger(__name__)) -> None:
    """Plot and save the ROC curve."""
    plot_roc(labels, probs, save_path=output_dir, show=False)
    logger.info(f"ROC curve saved to {os.path.join(output_dir, 'roc_curve.png')}")
