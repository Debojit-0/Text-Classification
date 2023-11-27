import os
from src.Textclassification.logging import logger
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer
from src.Textclassification.entity import ModelTrainerConfig
import pickle
import pandas as pd
import tensorflow as tf
from transformers import DistilBertTokenizer
from transformers import TFDistilBertForSequenceClassification
from transformers import TextClassificationPipeline
from transformers import TFDistilBertForSequenceClassification, TFTrainer, TFTrainingArguments
from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np



class ModelTrainer:
    def __init__(self, config:ModelTrainerConfig):
        self.config = config
        df = pd.read_csv(self.config.data_path)
        data_texts = df['text'].to_list()
        data_labels = df['encoded_text'].to_list()
        train_texts, val_texts, train_labels, val_labels = train_test_split(data_texts, data_labels, test_size = 0.2, random_state = 0 )
        train_texts, test_texts, train_labels, test_labels = train_test_split(train_texts, train_labels, test_size = 0.01, random_state = 0)
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        train_encodings = tokenizer(train_texts, truncation = True, padding = True  )

        val_encodings = tokenizer(val_texts, truncation = True, padding = True )

        class_names = ['sport', 'business', 'politics','tech', 'entertainment']

                
        """train_dataset = tf.data.Dataset.from_tensor_slices((
        dict(val_encodings),
        train_labels
        ))"""


        val_dataset = tf.data.Dataset.from_tensor_slices((
            dict(val_encodings),
            val_labels
        ))
        print("done 1")
        
        model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=5)

        


        training_args = TFTrainingArguments(
            output_dir=self.config.root_dir,          
            num_train_epochs=1,              
            per_device_train_batch_size=16,  
            per_device_eval_batch_size=64,   
            warmup_steps=500,                
            weight_decay=1e-5,  
            logging_dir='./logs',                       
            eval_steps=100                   
        )



        with training_args.strategy.scope():
            trainer_model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels = 5 )

 


        trainer = TFTrainer(
            model=trainer_model,                 
            args=training_args,                  
            train_dataset=val_dataset,         
            eval_dataset=val_dataset,

        
          
        )

        os.makedirs(self.config.data_path1, exist_ok=True)
        os.makedirs(self.config.data_path2, exist_ok=True)
        trainer.train()
        eval_results=trainer.evaluate()

        save_directory1 = self.config.data_path1
        save_directory2 = self.config.data_path2

         # Print the keys of eval_results
        print("Keys of eval_results:", eval_results.keys())
        print("Eval Results:", eval_results)

        predictions = model.predict(val_dataset)

        print(predictions)

    

        
       

        print("done 2")


        model.save_pretrained(save_directory1)

        tokenizer.save_pretrained(save_directory2)

      

                        
