from src.Textclassification.constants import *
from src.Textclassification.utils.common import read_yaml, create_directories
from src.Textclassification.entity import DataIngestionConfig
from src.Textclassification.entity  import DataCleaningConfig
from src.Textclassification.entity  import DataValidationConfig
from src.Textclassification.entity  import ModelTrainerConfig
from pathlib import Path
                                 

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

        print("Loaded Configuration:")
        print(self.config)
    
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir,
            root_dir1=config.root_dir1
            )

        return data_ingestion_config
   

    def get_data_cleaning_config(self) -> DataCleaningConfig:
        config = self.config.data_cleaning

        create_directories([config.root_dir])



        data_cleaning_config = DataCleaningConfig(
            root_dir=config.root_dir,
            data_files=config.data_files
        )

        return data_cleaning_config
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            ALL_REQUIRED_FILES=config.ALL_REQUIRED_FILES,
        )

        return data_validation_config

     

    def get_data_transformation_config(self) ->ModelTrainerConfig:
        config = self.config.model_trainer

        create_directories([config.root_dir])

        data_transformation_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            data_path1=config.data_path1,
            data_path2=config.data_path2,
            tokenizer_name = config.tokenizer_name
        )

        return data_transformation_config
    