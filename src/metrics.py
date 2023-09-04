"""
Module developed to compute model evaluation metrics and tensorboard syncs
"""

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score, precision_score
import pandas as pd
from plot import plot_confusion_matrix
import numpy as np


def compute_metrics(y_true, y_pred):
    """
    Args:
        - y_true: tensor containing ground truth values
        - y_pred: tensor containing predicted values
    Returns:
        - f1, accuracy, recall, precision metrics
    """
    labels = np.unique(y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', labels=labels)
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='macro', labels=labels)
    precision = precision_score(y_true, y_pred, average='macro', labels=labels)

    return {
        "F1 Score": f1,
        "Accuracy": accuracy,
        "Recall": recall,
        "Precision": precision
    }


def log_scalar_metrics_to_tensorboard(writer, epoch_n, metrics_dict):
    """
    Args:
        - writer: SummaryWriter object.
        - epoch_n: epoch number associated to the given values.
        - metrics_dict: dictionary containing computed metrics to log.
    """
    for metric in metrics_dict.keys():
        writer.add_scalar(metric, metrics_dict[metric], epoch_n)


def log_ce_loss_to_tensorboard(writer, epoch_n, loss):
    """
    Args:
        - writer: SummaryWriter object.
        - epoch_n: epoch number associated to the given value.
        - loss: cross-entropy loss value to log.
    """
    writer.add_scalar("Training Loss", loss, epoch_n)


def compute_confusion_matrix(y_true, y_pred, classes):
    """
    Args:
        - y_true: ground truth values
        - y_pred: predicted label values.
        - classes: classification labels.
    """
    cf_matrix = confusion_matrix(y_true.numpy(), y_pred.numpy())
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in classes],
                         columns=[i for i in classes])
    plot_confusion_matrix(df_cm)


def compute_average_metrics(f1_scores, accuracy_scores, recall_scores, precision_scores):
    """
    Args:
        - f1_scores: list containing all collected f1-score values
        - accuracy_scores: list containing all collected accuracy values
        - recall_scores: list containing all collected recall values
        - precision_scores: list containing all collected precision values

    Returns:
        - Map containing averages of metrics
    """
    return {
        "F1 Score": sum(f1_scores) / len(f1_scores),
        "Accuracy":  sum(accuracy_scores) / len(accuracy_scores),
        "Recall":  sum(recall_scores) / len(recall_scores),
        "Precision":  sum(precision_scores) / len(precision_scores),
    }
