import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion...")
        try:
            df = pd.read_csv(self.ingestion_config.raw_data_path)
            logging.info("Raw data loaded successfully.")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            logging.info("Creating train and test datasets...")
            df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
            df_train.to_csv(self.ingestion_config.train_data_path, index=False)
            df_test.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info("Data ingestion completed successfully.")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                # self.ingestion_config.raw_data_path
            )
        except Exception as e:
            logging.error(f"Error occurred during data ingestion: {e}")
            raise CustomException("Data ingestion failed.")