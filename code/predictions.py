import torch
import wandb
import numpy as np
import pandas as pd


class Predictor:
    def __init__(self, config):
        """
        Initializes an instance of the class.
        """
        self.config = config
        self.device = config["device"]

    @classmethod
    def from_dict(cls, cfg: dict):
        """
        Creates an Predictor object from a dictionary and data.

        Args:
            cfg (dict): A dictionary containing configuration parameters for the encoder.
            data: The data to be encoded.

        Returns:
            Predictor: An instance of the Predictor class.

        """
        config = cfg.get("model_parameters")
        return cls(config)

    def convert_labels_to_class(self, true_labels, pred_labels):
        """
        Converts the numeric class labels to their corresponding class names.

        Args:
            true_labels (list): List of true class labels.
            pred_labels (list): List of predicted class labels.

        Returns:
            true_labels (list): List of true class names.
            pred_labels (list): List of predicted class names.
        """
        classes = [
            "Data",
            "Documentation",
            "mission_instruments",
            "image",
            "Software_tools",
            "Training and Education"
        ]
        true_labels = [classes[i] for i in true_labels]
        pred_labels = [classes[i] for i in pred_labels]
        return true_labels, pred_labels

    def prediction(self, prediction_dataloader, model, links):
        """
        Performs predictions on a given dataset using the provided model.

        Args:
            prediction_dataloader (DataLoader): The dataloader for the prediction dataset.
            model: The trained model for making predictions.
            links (list): List of links associated with the prediction dataset.

        Returns:
            predictions (list): List of predicted probabilities for each class.
            true_labels (list): List of true class labels.
        """
        model.eval()
        # tracking variables
        predictions, true_labels = [], []
        for batch in prediction_dataloader:
            # Add batch to CPU
            batch = tuple(t.to(self.device) for t in batch)

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients, saving memory and
            # speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                result = model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    return_dict=True,
                )
            logits = result.logits
            logits = torch.sigmoid(logits)
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            b_labels = b_labels.to("cpu").numpy()
            # Store predictions and true labels
            predictions.append(logits)
            true_labels.append(b_labels)
        pred_flat = [np.argmax(arr, axis=1).tolist() for arr in predictions]
        pred_flat = [i for lis in pred_flat for i in lis]
        labels_flat = [i.tolist() for i in true_labels]
        labels_flat = [i for lis in labels_flat for i in lis]
        labels_flat, pred_flat = self.convert_labels_to_class(labels_flat, pred_flat)
        table = pd.DataFrame(
            {"urls": links, "predictions": pred_flat, "truth": labels_flat}
        )
        table.to_csv("test.csv")
        table = wandb.Table(dataframe=table)
        # Log the table to the run
        wandb.log({"predictions": table})
        return predictions, true_labels
