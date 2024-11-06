import os
import json
import pandas as pd
import pycaret
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
import mlflow.pyfunc
from src.read_data_source import DataIngestor 
from src.preprocess_data import DataPreprocessing
from src.training import ModelTraining
from pycaret.classification import setup, create_model, tune_model, compare_models, pull,evaluate_model,save_model
from datetime import datetime



class Pipeline:
    def __init__(self):
        pass

    def run_pipeline(self,config_folder_path ,data_source_config_pattern,preprocess_config_pattern,training_config_pattern,model_mapping_config_path,model_mapping_pattern,exp_name):
    
        # Create an instance of the DataIngestor class and load the datasource config JSON
        data_ingestor = DataIngestor()
        dataset= data_ingestor.get_dataset_from_source(config_folder_path,data_source_config_pattern)
        dataset.head()
        print(dataset.info())
        # Create an instance of the DataPreprocessing class
        data_preprocessor = DataPreprocessing()
        preprocessed_dataset=data_preprocessor.get_preprocessed_dataset(dataset,config_folder_path,preprocess_config_pattern)
        print("------------------------------------------------")
        print(preprocessed_dataset.shape)
        print(preprocessed_dataset.info())
        print("------------------------------------------------")
        # # Create an instance of the ModelTraining class
        model_trainer= ModelTraining()
        MLFLOW_TRACKING_URI = "mlflowhub://radhika_halde"
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(exp_name)
        model_trainer.train_and_save_model(preprocessed_dataset,config_folder_path,training_config_pattern,model_mapping_config_path,model_mapping_pattern,exp_name)    
        

if __name__ == "__main__":
    pipeline = Pipeline()
    exp_name = "dellwb_experiment_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    print(exp_name) 
    #changes for c5
    config_folder_path = "../12345-20241106205058"
    data_source_config_pattern = "datasource_config.json"
    preprocess_config_pattern = "preprocessing_config.json" 
    training_config_pattern = "model_training_config.json"  
    model_mapping_config_path = "config_folder"
    model_mapping_pattern = "model_mapping.json"
    pipeline.run_pipeline(config_folder_path,data_source_config_pattern,preprocess_config_pattern,training_config_pattern,model_mapping_config_path,model_mapping_pattern,exp_name)
    


    