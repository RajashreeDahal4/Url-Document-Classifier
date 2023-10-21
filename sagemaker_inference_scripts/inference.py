import os
import pandas as pd

import torch.nn as nn
import json
import inspect
from model import ModelBert
from test_predictions import TestPredictor
from load_dataset import DataLoad
import multiprocessing
from preprocessing import Preprocessor
from encoder import Encoder


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



def model_fn(model_dir):
    current_file_path = inspect.getfile(lambda: None)
    config_file_path='/opt/ml/model/code/config.json'
    with open(config_file_path, "rb") as files:
        config = json.load(files)
    loaded_model=ModelBert.from_dict(config)
    model = loaded_model.load_model(model_dir)
    return model

def predict_fn(input_data, model):
    prediction = {}
    encoded_data, pdf_lists, image_lists,predictor,config,collection_id=input_data
    if len(encoded_data) > 0:
        input_ids, attention_masks, links = predictor.tokenize_test_data(encoded_data)
        loader = DataLoad.from_dict(config)
        loader.dataset(input_ids, attention_masks)
        inference_dataloader = loader.dataloader()
        category = predictor.predict_test_data(inference_dataloader,model)
        for enum, each_category in enumerate(category):
            prediction[links[enum]] = config.get("webapp").get(each_category)
    for image_url in image_lists:
        prediction[image_url] = config.get("webapp").get("image")
    for pdf_url in pdf_lists:
        prediction[pdf_url] = config.get("webapp").get("Documentation")
    outputs=(prediction,pdf_lists)
    return  {"out":outputs,"collection_id":collection_id}
    

def output_fn(prediction, content_type):
    if content_type == 'application/json':
        return json.dumps(prediction)
    raise Exception('Unsupported Content Type')

def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        urls_dict = json.loads(request_body)
        urls=urls_dict['urls']
        collection_id=urls_dict['collection_id']
        config_file_path='/opt/ml/model/code/config.json'
        with open(config_file_path, "rb") as files:
            config = json.load(files)
        predictor = TestPredictor.from_dict(config)
        prediction = {}
        pdf_lists,image_lists,encoded_data=multiprocess_process_data(urls,config)
        input_data=(encoded_data,pdf_lists,image_lists,predictor,config,collection_id)
        return input_data
    else:
        raise Exception('Unsupported Content Type')