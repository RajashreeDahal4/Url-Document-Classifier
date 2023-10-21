import importlib
import random
import pandas as pd
import numpy as np
import torch
from encoder import Encoder
from preprocessing import Preprocessor
import pickle

random.seed(42)


class Tokenizer:
    '''
    class for tokenizing the input
    '''
    def __init__(self, config):
        """
        Initializes a Tokenizer object.

        Args:
            data: The data to be encoded.
            config (dict): A dictionary containing configuration parameters for the encoder.

        """
        data_path = config["preprocess"]["processed_data"]
        self.data = pd.read_csv(data_path)
        self.config = config
        self.final_data = pd.DataFrame(columns=["text", "class", "links"])
        self.train_idx = []
        self.test_idx = []
        self.val_idx = []

    @classmethod
    def from_dict(cls, cfg: dict):
        """
        Creates an Encoder object from a dictionary and data.

        Args:
            cfg (dict): A dictionary containing configuration parameters for the encoder.
            data: The data to be encoded.

        Returns:
            Encoder: An instance of the Encoder class.

        """
        return cls(cfg)

    def train_test_split(self):
        """
        Performs the train-test split on the data, preprocesses the features,
        encodes the data, and sets the indices for the training, validation, and
        test sets.

        Returns:
            None
        """
        # Randomly shuffle the rows
        self.data = self.data.sample(frac=1, random_state=42)
        # Reset the index of the shuffled DataFrame
        self.data = self.data.reset_index(drop=True)
        # self.data=self.data[0:30]
        # links = self.data.links.values
        processor = Preprocessor.from_dict(self.config, self.data)
        dataframe = processor.preprocessed_features()
        # Specify the columns you want to select
        self.final_data = pd.DataFrame(
            {
                "text": dataframe["soup"],
                "class": dataframe["class"],
                "links": dataframe["links"],
            }
        )
        encoder = Encoder.from_dict(self.config, self.final_data)
        self.data = encoder.encoder()
        self.data["final_sentence"] = self.data["text"]
        self.data = self.data.drop(columns=["text"])
        # First, calculate the split sizes. 80% training, 10% validation, 10% test.
        train_size = int(self.config["preprocess"]["train_size"] * len(self.data))
        val_size = int(self.config["preprocess"]["valid_size"] * len(self.data))
        # Create a list of indeces for all of the samples in the dataset.
        indeces = np.arange(0, len(self.data))
        # Find indices of instances from class "5"
        # Shuffle the indeces randomly.
        random.shuffle(indeces)
        # Find indices of instances from class "5"
        class_5_indices = [idx for idx in indeces if self.data["class"][idx] == 5]
        # Separate class 5 indices from the rest
        other_indices = [idx for idx in indeces if idx not in class_5_indices]
        # Get a list of indeces for each of the splits.
        # Ensure there's at least one instance of class "5" in each split
        class_5_train_size=int(len(class_5_indices)*self.config["preprocess"]["train_size"])
        class_5_valid_size=int(len(class_5_indices)*self.config["preprocess"]["valid_size"])
        self.train_idx = class_5_indices[:class_5_train_size-1] + other_indices[:train_size - 1]
        self.val_idx = class_5_indices[class_5_train_size-1:class_5_train_size+class_5_valid_size-2] + other_indices[train_size - 1:train_size + val_size - 2]
        self.test_idx = class_5_indices[class_5_train_size+class_5_valid_size-2:] + other_indices[train_size + val_size - 2:]



    def token(self):
        """
        Tokenizes the sentences, encodes them using tokenizer, and prepares the
        input tensors and labels for training.
        Returns:
            input_ids (torch.Tensor): Tensor of input token IDs for each sentence.
            attention_masks (torch.Tensor): Tensor of attention masks for each sentence.
            labels (torch.Tensor): Tensor of labels for each sentence.
            train_idx (ndarray): Indices for the training set.
            val_idx (ndarray): Indices for the validation set.
            test_idx (ndarray): Indices for the test set.
            links (list): List of links associated with each sentence.
        """
        self.train_test_split()
        sentence, labels, links, input_ids, attention_masks = [], [], [], [], []
        for _, row in self.data.iterrows():
            sentence.append(row["final_sentence"])
            labels.append(row["class"])
            links.append(row["links"])

        module_name = self.config["model_parameters"]["module_name"]
        transformers = importlib.import_module(module_name)
        # Dynamically get the model class from transformers module
        tokenizer_class = getattr(
            transformers, self.config["model_parameters"]["tokenizer"]
        )
        tokenizer = tokenizer_class.from_pretrained(
            self.config["model_parameters"]["model_type"]
        )

        for sent in sentence:
            encoded_dict = tokenizer.encode_plus(
                text=sent,
                add_special_tokens=True,
                truncation=True,
                max_length=500,
                padding="max_length",
                return_attention_mask=True,
                return_tensors="pt",
            )
            input_ids.append(encoded_dict["input_ids"])
            attention_masks.append(encoded_dict["attention_mask"])
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels)
        token_results = {
            "input_ids": input_ids,
            "attention_masks": attention_masks,
            "labels": labels,
            "train_idx": self.train_idx,
            "val_idx": self.val_idx,
            "test_idx": self.test_idx,
            "links": links,
        }
        return token_results
