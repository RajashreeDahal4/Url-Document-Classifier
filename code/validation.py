import torch
from metrics import flat_accuracy
import wandb


class Validator:
    """
    class for validation dataset
    """

    def __init__(self, model, validation_dataloader, device):
        self.model = model
        self.validation_dataloader = validation_dataloader
        self.device = torch.device(device)

    @classmethod
    def from_dict(cls, cfg, model, validation_dataloader):
        """
        Creates an Validator object from a dictionary.

        Args:
            cfg (dict): A dictionary containing configuration parametersr.
            model: the trained loader
            validation_dataloader: dataloader of validation dataset

        Returns:
            Validator:  An instance of validator class
        """
        model_parameters = cfg.get("model_parameters")
        return cls(model, validation_dataloader, device=model_parameters.get("device"))

    def validation(self):
        """
        Performs validation on the model using the validation dataset.

        Returns:
            tuple: A tuple containing the predicted logits and true labels of the
            validation dataset.

        """
        self.model.eval()
        total_eval_accuracy, total_eval_loss, nb_eval_steps = 0, 0, 0
        val_prediction, val_true_labels = [], []
        # Evaluate data for one epoch
        for batch in self.validation_dataloader:
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)

            with torch.no_grad():
                result = self.model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels,
                    return_dict=True,
                )
            # Get the loss and "logits" output by the model. The "logits" are the
            # output values prior to applying an activation function like the
            # softmax.
            loss = result.loss
            logits = result.logits

            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            logits = torch.sigmoid(logits)
            logits = logits.detach().cpu().numpy()
            b_labels = b_labels.to("cpu").numpy()

            val_prediction.append(logits)
            val_true_labels.append(b_labels)
            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            accuracy = flat_accuracy(logits, b_labels)
            total_eval_accuracy += accuracy
            nb_eval_steps += 1
        # Report the final accuracy for this validation run.
        validation_accuracy = total_eval_accuracy / nb_eval_steps
        print(f" Validation Accuracy: {validation_accuracy:.2f}")
        wandb.log({"avg validation_accuracy": validation_accuracy})

        # Calculate the average loss over all of the batches.
        print(f"  Validation Loss: {(total_eval_loss / nb_eval_steps):.2f}")
        wandb.log({"avg validation loss": total_eval_loss / nb_eval_steps})
        return val_prediction, val_true_labels
