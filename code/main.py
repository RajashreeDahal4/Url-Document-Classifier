import argparse
import json
from load_dataset import DataLoad
from metrics import metrics, print_metrics
from model import ModelBert
from train import Trainer
from predictions import Predictor
from dataset import convert_dataset_to_csv
from tokenizer import Tokenizer
from test_predictions import TestPredictor
import wandb


def predict(prediction_dataloader, test_links, config):
    """
    This function makes predictions on test documents using a trained document tagging model.

    Parameters:
    - prediction_dataloader: The dataloader containing the input data for prediction.
    - test_links: A list of links to test documents.
    - config: The configuration dictionary containing various settings for prediction.
    """

    trained_model = ModelBert.from_dict(config)
    model = trained_model.load_model()
    model.eval()
    predictor = Predictor.from_dict(config)
    pred, labels = predictor.prediction(prediction_dataloader, model, test_links)
    performance_metrics = metrics(pred, labels, config["classes"])
    print_metrics(performance_metrics, mode="test")


def train(config_file):
    """
    This function trains a document tagging model using the provided configuration.
    Returns:
    None
    """
    wandb.init(project="Automated Document Tagging2")
    with open(config_file, encoding="utf-8") as files:
        config = json.load(files)
    convert_dataset_to_csv(config)
    tokenizer = Tokenizer.from_dict(config)
    token_results = tokenizer.token()
    loader = DataLoad.from_dict(config)
    loader.dataset(token_results)
    (
        train_dataloader,
        validation_dataloader,
        prediction_dataloader,
        test_links,
    ) = loader.dataloader()
    trainer = Trainer.from_dict(config)
    pred, labels = trainer.train(train_dataloader, validation_dataloader)
    performance_metrics = metrics(pred, labels, config["classes"])
    print_metrics(performance_metrics, mode="validation")
    predict(prediction_dataloader, test_links, config)


def predicts(config_file, url):
    """
    Predicts the possible full_forms and their confidence_scores which exceed the
    confidence_threshold given context and full_forms as input
    Arg(s):
            config_file: json file for config
            url (str): The URL of the test data
    Returns:
        str: The predicted category of the test data.
    """
    with open(config_file, "rb") as files:
        config = json.load(files)
    predictor = TestPredictor.from_dict(config)
    encoded_data = predictor.process_test_data(url)
    if type(encoded_data) == str:
        print("Image")
        return "Image"
    input_ids, attention_masks = predictor.tokenize_test_data(encoded_data)
    category,confidence_score = predictor.predict_test_data(input_ids, attention_masks)
    return category,confidence_score

def batch_predicts(config_file, urls):
    """
    Predicts the possible full_forms and their confidence_scores which exceed the
    confidence_threshold given context and full_forms as input in a batch of maximum 8 urls
    Arg(s):
            config_file: json file for config
            urls (list): The URL of the test data in the form of list
    Returns:
        list: The predicted category of the test data for each url in the form of list.
    """
    with open(config_file, "rb") as files:
        config = json.load(files)
    predictor = TestPredictor.from_dict(config)
    prediction={}
    url_list=[]
    count=0
    for url in urls:
        count=count+1
        encoded_data=predictor.process_test_data(url)
        if type(encoded_data) == str:
            print("Image")
            prediction['url']={'category':"Image","confidence score":100}
        else:
            url_list.append(url)
            input_ids, attention_masks = predictor.tokenize_test_data(encoded_data)
            category,confidence_score = predictor.predict_test_data(input_ids, attention_masks)
            prediction[url]={"category":category,"confidence score":str(confidence_score*100)}
    return prediction

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training \
                and prediction with given configuration file."
    )
    subparsers = parser.add_subparsers(dest="subparser_name", help="sub-command help")

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--config_file", type=str, help="Path of the configuration file."
    )

    pred_parser = subparsers.add_parser(
        "predicts", help="Make predictions using the model"
    )
    pred_parser.add_argument(
        "--config_file", type=str, help="Path of the configuration file."
    )
    pred_parser.add_argument("--url", type=str, help="url link")

    args = parser.parse_args()

    if args.subparser_name == "train":
        train(args.config_file)
    if args.subparser_name == "predicts":
        predicts(args.config_file, args.url)
    else:
        parser.print_help()
