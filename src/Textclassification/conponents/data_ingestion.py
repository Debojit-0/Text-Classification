import os
import urllib.request as request
import zipfile
from src.Textclassification.logging import logger
from src.Textclassification.utils.common import get_size
from pathlib import Path
from src.Textclassification.entity import DataIngestionConfig
import pandas as pd
from sklearn.model_selection import train_test_split


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config


    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url = self.config.source_URL,
                filename = self.config.local_data_file
            )
            logger.info(f"{filename} download! with following info: \n{headers}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")  

    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)

    def split_data(self, test_size=0.2, validation_size=0.1, random_state=42):
        """
        Split the data into train, test, and validation sets.

        Parameters:
        - test_size: The proportion of the dataset to include in the test split.
        - validation_size: The proportion of the dataset to include in the validation split.
        - random_state: Seed for the random number generator for reproducibility.

        Returns:
        - train_data: DataFrame, training set
        - test_data: DataFrame, testing set
        - validation_data: DataFrame, validation set
        """
        # Assuming that your data is in a CSV file, modify the following line accordingly
        data = pd.read_csv(os.path.join(self.config.unzip_dir, 'bbc-text.csv'))

       

        data.to_csv(os.path.join(self.config.unzip_dir, 'data.csv'), index=False)
      

        return data
    


        
    
