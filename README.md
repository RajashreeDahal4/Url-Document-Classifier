# Automated Document Tagging


# Project Description: 
The purpose of this project is to classify the content of a given url into one of the six classes "Image","Documentation","Software and Tools", "Mission and Instruments", "Data", and "Training and Education". 

#Datasets:
Reference link for datasets: https://docs.google.com/spreadsheets/d/1rK7hvb_HRd-sqL3jrSYll5BiDvwnzQY2qVWDmpg6Bbk/edit#gid=1560325588

# to run the repository:
* python3 -m venv venv
* source venv/bin/activate
* pip install -r requirements.txt
* location for saved model in drive: https://drive.google.com/drive/u/1/folders/1jkJSpN3ZuXhZIis4dSc-v0LkSV3pMrcs
* saved weight_name: model.pt
* train the model: python3 main.py train --config_file config.json
* prediction sample:python3 main.py predicts --config_file config.json --url "url_link"



For more details: contact rd0081@uah.edu
