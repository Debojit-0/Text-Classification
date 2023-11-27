from dataclasses import dataclass
from pathlib import Path
from typing import Dict

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    root_dir1: Path


@dataclass(frozen=True)
class DataCleaningConfig:
    root_dir: Path
    data_files:  Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    data_path: Path
    tokenizer_name: Path
    data_path1: Path
    data_path2: Path



    



