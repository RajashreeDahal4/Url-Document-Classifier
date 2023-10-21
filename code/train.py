import importlib

# from transformers import AdamW
# from transformers import get_linear_schedule_with_warmup
import torch
from validation import Validator
import wandb
from model import ModelBert


class Trainer:
    '''
    class for training the model
    '''
    def __init__(
        self,
        config,
        learning_rate=0.00001,
        epochs=9,
    ):
        """
        Initializes an instance of the class.

        Args:
            config (dict): A dictionary containing the configuration parameters for the model.
            learning_rate (float, optional): The learning rate for the optimizer.
            epochs (int, optional): The number of training epochs.
            batch_size (int, optional): The batch size for training.
            model_path (str, optional): The path to save the trained model.".
        """
        self.config = config
        self.learning_rate = learning_rate
        self.epochs = epochs
        model = ModelBert.from_dict(self.config)
        self.model, _ = model.make_model()
        self.device = torch.device("cpu")
        module_name = self.config["model_parameters"]["module_name"]
        self.transformers = importlib.import_module(module_name)
        optimizer_class = getattr(
            self.transformers, self.config["model_parameters"]["optimizer"]
        )
        self.optimizer = optimizer_class(
            self.model.parameters(), lr=self.learning_rate, eps=1e-8
        )

    @classmethod
    def from_dict(cls, cfg: dict):
        """
        Creates an Trainer object from a dictionary.

        Args:
            cfg (dict): A dictionary containing configuration parametersr.
            data: The data to be encoded.

        Returns:
            Trainer: An instance of the Trainer class.
        """
        model_parameters = cfg.get("model_parameters")

        return cls(
            cfg,
            learning_rate=model_parameters.get("learning_rate"),
            epochs=model_parameters.get("epochs"),
        )

    def train(self, train_dataloader, validation_dataloader):
        """
        Trains the model on the provided training dataset and validates it using
        the validation dataset.

        Args:
            train_dataloader (DataLoader): The dataloader for the training dataset.
            validation_dataloader (DataLoader): The dataloader for the validation dataset.

        Returns:
            pred (list): The predicted values from the validation dataset.
            labels (list): The actual labels from the validation dataset.
        """
        total_steps = len(train_dataloader) * self.epochs
        scheduler_class = getattr(
            self.transformers, self.config["model_parameters"]["scheduler"]
        )
        scheduler = scheduler_class(
            self.optimizer,
            num_warmup_steps=0,  # Default value in run_glue.py
            num_training_steps=total_steps,
        )
        for epoch_i in range(0, self.epochs):
            print(f"======== Epoch {epoch_i + 1} / {self.epochs} ========")

            print("Training...")
            total_train_loss = 0
            self.model.train()
            num_train_steps = 0
            for _, batch in enumerate(train_dataloader):
                self.model.zero_grad()

                result = self.model(
                    batch[0].to(self.device),
                    token_type_ids=None,
                    attention_mask=batch[1].to(self.device),
                    labels=batch[2].to(self.device),
                    return_dict=True,
                )

                loss = result.loss

                total_train_loss += loss.item()
                loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                scheduler.step()
                num_train_steps += 1
            # Calculate the average loss over all of the batches.
            train_loss = total_train_loss / num_train_steps
            print("train loss", train_loss)
            wandb.log({"avg train_loss": train_loss})

            validator = Validator(
                self.model,
                validation_dataloader,
                self.config["model_parameters"]["device"],
            )
            pred, labels = validator.validation()
        torch.save(
            self.model.state_dict(), self.config["model_parameters"]["saved_model_name"]
        )
        return pred, labels
