import sys
from dataclasses import dataclass

from scipy.sparse import hstack

from geopy.distance import great_circle

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_function, feature_engg

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')
            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_columns = ['city', 'job', 'gender', 'merchant', 'category']
            numerical_columns = ['cc_freq', 'cc_freq_class', 'age', 'distance_km', 'month', 'day','hour', 'hours_diff_bet_trans', 'amt']
            
            logging.info('Pipeline Initiated')

            ## Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="mean")), 
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            # Categorigal Pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")), 
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns), 
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )
            
            return preprocessor

            logging.info('Pipeline Completed')

        except Exception as e:
            logging.info("Error in Data Trnasformation")
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            
            #Feature Engineering for train data
            train_df = feature_engg(train_df)

            #Feature Engineering for test data
            test_df = feature_engg(test_df)
            

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            train_df = train_df[['cc_freq','cc_freq_class','city','job','age','gender','merchant', 'category','distance_km','month','day','hour','hours_diff_bet_trans','amt','is_fraud', 'Unnamed: 0','street','state','first','last','trans_num','unix_time', 'dob', 'lat','long','merch_lat','merch_long', 'cc_num','trans_date_trans_time','city_pop']]
            test_df = test_df[['cc_freq','cc_freq_class','city','job','age','gender','merchant', 'category','distance_km','month','day','hour','hours_diff_bet_trans','amt','is_fraud', 'Unnamed: 0','street','state','first','last','trans_num','unix_time', 'dob', 'lat','long','merch_lat','merch_long', 'cc_num','trans_date_trans_time','city_pop']]

            target_column_name = 'is_fraud'
            drop_columns = [target_column_name,'Unnamed: 0','street','state','first','last','trans_num','unix_time', 'dob', 'lat','long','merch_lat','merch_long', 'cc_num','trans_date_trans_time','city_pop']

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]

            logging.info(f'Train Dataframe Head after Feature Engineering: \n{input_feature_train_df.head().to_string()}')

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]
            logging.info(f'Test Dataframe Head after Feature Engineering: \n{input_feature_test_df.head().to_string()}')

            
            ## Trnasformating using preprocessor obj
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")
            
            input_feature_train_dtype = input_feature_train_arr.dtype
            target_feature_train_dtype = target_feature_train_df.dtype

            target_feature_train_df = target_feature_train_df.astype(input_feature_train_dtype)

            input_feature_test_dtype = input_feature_test_arr.dtype
            target_feature_test_dtype = target_feature_test_df.dtype
            target_feature_test_df = target_feature_test_df.astype(input_feature_test_dtype)
           
            # Convert target_feature_train_df to a NumPy array and reshape it
            target_train_arr = target_feature_train_df.to_numpy().reshape(-1, 1)

            # Concatenate input_feature_train_arr and target_train_arr horizontally
            train_arr = hstack((input_feature_train_arr, target_train_arr))

            # Convert target_feature_test_df to a NumPy array and reshape it
            target_test_arr = target_feature_test_df.to_numpy().reshape(-1, 1)

            # Concatenate input_feature_test_arr and target_test_arr horizontally
            test_arr = hstack((input_feature_test_arr, target_test_arr))



            save_function(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)