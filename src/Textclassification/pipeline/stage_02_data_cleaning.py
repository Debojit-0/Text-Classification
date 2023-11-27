from src.Textclassification.conponents.data_cleaning import DataCleaning
from src.Textclassification.config.configuration import ConfigurationManager
from src.Textclassification.logging import logger


class DataCleaningTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_cleaning_config = config.get_data_cleaning_config() 
        data_cleaning = DataCleaning(config=data_cleaning_config)
        data_cleaning.clean_data()