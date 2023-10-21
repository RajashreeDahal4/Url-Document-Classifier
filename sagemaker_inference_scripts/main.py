import json

from load_dataset import DataLoad
from test_predictions import TestPredictor

import json

from load_dataset import DataLoad
from test_predictions import TestPredictor
import multiprocessing
from preprocessing import Preprocessor
import pandas as pd
import numpy as np
from encoder import Encoder
import torch 
import logging
logger = logging.getLogger(__name__)
from model import ModelBert

import multiprocessing


def model_fn(model_dir,config):
    loaded_model=ModelBert.from_dict(config)
    model = loaded_model.load_model("xlnet_model.pt")
    return model


def process_test_data(parameter):
    """
    Processes the test data by retrieving content from the provided URL and encoding it.

    Parameters:
        url (str): The URL of the test data.

    Returns:
        Union[str, DataFrame]: If the content type is an image, returns "Image".
                            Otherwise, returns the encoded test data as a DataFrame.

    """
    urls,config=parameter
    dataframe=pd.DataFrame()
    dataframe["links"] = urls
    dataframe["class"] = [3 for i in urls]  # any random class
    processor = Preprocessor.from_dict(config, dataframe)
    (
        dataframe,
        pdf_lists,
        image_lists,
    ) = processor.preprocessed_features()
    dataframe["text"] = dataframe["soup"]
    encoder = Encoder.from_dict(config, dataframe)
    encoded_data = encoder.encoder()
    return encoded_data, pdf_lists, image_lists



def multiprocess_process_data(urls,config):
    pdf_lists,image_lists=[],[]
    encoded_data = pd.DataFrame()
    # Split the list of URLs into chunks for processing
    num_processes = multiprocessing.cpu_count()
    num_processes=10
    # Ensure num_processes is at most equal to the number of URLs
    num_processes = min(num_processes, len(urls))
    chunk_size = len(urls) // num_processes
    url_chunks = [urls[i:i + chunk_size] for i in range(0, len(urls), chunk_size)]
    # Create a pool of worker processes
    pool = multiprocessing.Pool(num_processes)
    results = pool.map(process_test_data, [(chunk,config) for chunk in url_chunks])
    pool.close()
    pool.join()
    for chunk_encoded_data, chunk_pdf_lists,chunk_image_lists in results:
        encoded_data = pd.concat([encoded_data,chunk_encoded_data], ignore_index=True)
        image_lists.extend(chunk_image_lists)
        pdf_lists.extend(chunk_pdf_lists) 
    return pdf_lists,image_lists,encoded_data


def batch_predicts(config_file, urls):
    """
    Predicts category of each url given a list of urls
    Arg(s):
            config_file: json file for config
            urls (list): The URL of the test data in the form of list
    Returns:
        tuple: Tuple of dictionary with key as url and its value as class category, and lists of url with pdf url_type
    """
    if __name__ == '__main__':
        with open(config_file, "rb") as files:
            config = json.load(files)
        loaded_model=model_fn("xyz",config)
        predictor = TestPredictor.from_dict(config)
        prediction = {}
        pdf_lists,image_lists,encoded_data=multiprocess_process_data(urls,config)
        if len(encoded_data) > 0:
            input_ids, attention_masks, links = predictor.tokenize_test_data(encoded_data)
            loader = DataLoad.from_dict(config)
            loader.dataset(input_ids, attention_masks)
            inference_dataloader = loader.dataloader()
            category = predictor.predict_test_data(inference_dataloader,loaded_model)
            for enum, each_category in enumerate(category):
                prediction[links[enum]] = config.get("webapp").get(each_category)
        for image_url in image_lists:
            prediction[image_url] = config.get("webapp").get("image")
        for pdf_url in pdf_lists:
            prediction[pdf_url] = config.get("webapp").get("Documentation")
        return prediction, pdf_lists





if __name__ == '__main__':
    result = batch_predicts("config.json", ["https://simplex.giss.nasa.gov/gcm/ROCKE-3D/", "https://simplex.giss.nasa.gov/gcm/ROCKE-3D/GISS_ROCKE-3D_tutorial_SLIDES.pdf", ...])
    print("result", result)
