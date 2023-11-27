from src.Textclassification.conponents.model_trainer import ModelTrainer
from src.Textclassification.config.configuration import ConfigurationManager
from src.Textclassification.logging import logger


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        #data_cleaning_config = config.get_data_transformation_config() 
        model_trainer_config = config.get_data_transformation_config()
        model_trainer = ModelTrainer(config=model_trainer_config)  