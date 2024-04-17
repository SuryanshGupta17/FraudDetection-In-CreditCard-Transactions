import pandas as pd 
import numpy as np 
import sys 
import os
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet 
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from src.logger import logging 
from src.exception import CustomException 
from dataclasses import dataclass
from src.utils import save_function 
from src.utils import model_performance 

@dataclass 
class ModelTrainerConfig():
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer():
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array): 
        try: 
            logging.info("Seggregating the dependent and independent variables")
            # Convert train_array and test_array to CSR format
            train_array = train_array.tocsr()
            test_array = test_array.tocsr()

            # Slice the CSR matrices to get X_train, y_train, X_test, y_test
            X_train, y_train = train_array[:, :-1], train_array[:, -1].toarray().ravel()
            X_test, y_test = test_array[:, :-1], test_array[:, -1].toarray().ravel()

            models = {
                'Logistic Regression': LogisticRegression(random_state=10),
                'Decision Tree': DecisionTreeClassifier(random_state=10)
            }
            model_report: dict = model_performance(X_train, y_train, X_test, y_test, models)

            print(model_report)
            print("\n"*100)
            logging.info(f"Model Report: {model_report}")

            # Best model
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)] # Study again
            
            best_model = models[best_model_name]

            print(f"The best model is {best_model_name}, with Accuracy Score: {best_model_score}")
            print("\n"*100)
            logging.info(f"The best model is {best_model_name}, with Accuracy Score: {best_model_score}")
            save_function(file_path= self.model_trainer_config.trained_model_file_path, obj = best_model)


        except Exception as e: 
            logging.info("Error occured during model training")
            raise CustomException(e,sys)