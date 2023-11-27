import pandas as pd
from pathlib import Path
from src.Textclassification.entity import DataCleaningConfig

class DataCleaning:
    def __init__(self, config: DataCleaningConfig):
        self.config = config

    def clean_data(self):
        data_path = self.config.data_files  # Directly access the Path object

        # Read the data file
        data_df = pd.read_csv(data_path)

        # Cleaning steps
        # Add a new column 'count' representing the word count in each text
        data_df['count'] = data_df['text'].apply(lambda x: len(x.split()))

        # Encode the 'category' column and create a new 'encoded_text' column
        data_df['encoded_text'] = data_df['category'].astype('category').cat.codes

        print(data_df)

        # Save the cleaned data with the specified file name in the same location as the root directory
        cleaned_data_path = Path(self.config.root_dir, 'cleaned_data.csv')
        data_df.to_csv(cleaned_data_path, index=False)

        return data_df  # Return the cleaned DataFrame without converting to lists
