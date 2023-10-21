import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb


def flat_accuracy(preds, labels):
    """
    Calculates the accuracy of the predictions by comparing them with the ground truth labels.
    Args:
        preds (numpy.ndarray): Array containing model predictions.
        labels (numpy.ndarray): Array containing ground truth labels.
    Returns:
        The accuracy of the predictions as a float.
    """
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def classwise_accuracy(preds, labels, classes):
    """
    This function calculates class-wise accuracy and generates a confusion matrix,
    classification report
    Parameters:
    - preds: Predicted labels.
    - labels: True labels.
    - classes: Dictionary containing class labels as keys and corresponding class names as values.
    """
    confusion__matrix = confusion_matrix(labels, preds)
    # Plot confusion matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion__matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=list(classes.keys()),
        yticklabels=list(classes.keys()),
    )
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    image = wandb.Image(plt)
    wandb.log({"confusion_matrix": image})

    # Calculate class-wise accuracy
    class_accuracies = {}
    for i in range(confusion__matrix.shape[0]):
        class_label = list(classes.keys())[i]
        class_total = confusion__matrix[i].sum()
        class_correct = confusion__matrix[i, i]
        accuracy = class_correct / class_total
        class_accuracies[class_label] = accuracy
    print("class-accuracies", class_accuracies)
    accuracies = pd.DataFrame(
        list(class_accuracies.items()), columns=["Class", "Accuracy"]
    )
    accuracy_table = wandb.Table(dataframe=accuracies.set_index("Class"))
    wandb.log({"class-accuracies": accuracy_table})

    report = classification_report(
        labels, preds, target_names=list(classes.keys()), output_dict=True
    )
    # Convert classification report to DataFrame
    df_report = pd.DataFrame(report).transpose()
    wandb.log({"classification report": wandb.Table(dataframe=df_report)})


def metrics(preds, labels, classes):
    """
    This function calculates performance metrics for a multi-class classification task.
    Parameters:
    - preds: Predicted labels or probabilities from the model. It should be a list
    of arrays or lists.
    - labels: True labels for the corresponding predictions. It should be a list of arrays or lists.
    - classes: List of class labels.
    Returns:
    A dictionary containing the calculated performance metrics.
    """
    performance_metrics = {}
    pred_flat = [np.argmax(arr, axis=1).tolist() for arr in preds]
    pred_flat = [i for lis in pred_flat for i in lis]
    labels_flat = [i.tolist() for i in labels]
    labels_flat = [i for lis in labels_flat for i in lis]
    classwise_accuracy(pred_flat, labels_flat, classes)

    performance_metrics["accuracy"] = np.sum(
        np.array(pred_flat) == np.array(labels_flat)
    ) / len(labels_flat)
    performance_metrics["f1score"] = f1_score(labels_flat, pred_flat, average="micro")
    precision, recall, _, _ = precision_recall_fscore_support(
        labels_flat, pred_flat, average="micro"
    )
    performance_metrics["precision"] = precision
    performance_metrics["recall"] = recall
    return performance_metrics


def print_metrics(result, mode):
    """
    Prints metrics evaluated and logs the information in wandb
    Args: Dictionary
    """
    recall = result["recall"]
    precision = result["precision"]
    f1score = result["f1score"]
    accuracy = result["accuracy"]
    print(f"{mode} Precision: {precision}")
    print(f"{mode} recall: {recall}")
    print(f"{mode} f1_score: {f1score}")
    print(f"{mode} accuracy: {accuracy}")
    wandb.log({mode + "precision": precision})
    wandb.log({mode + "recall": recall})
    wandb.log({mode + "f1_score": f1score})
    wandb.log({mode + "accuracy": accuracy})
