import sys
from dataclasses import dataclass
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
from src.utils import save_function

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
            numerical_columns = ['cc_freq', 'cc_freq_class', 'age', 'distance_km', 'month', 'day', 'hour', 'hours_diff_bet_trans', 'amt']
            
            
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
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'is_fraud'
            drop_columns = [target_column_name,'id', 'Unnamed: 0','street','state','first','last','trans_num','unix_time']

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]


            input_feature_train_df['trans_date_trans_time'] = pd.to_datetime(input_feature_train_df['trans_date_trans_time'],format='mixed')

            input_feature_train_df['hour'] = input_feature_train_df['trans_date_trans_time'].dt.hour
            input_feature_train_df['day'] = input_feature_train_df['trans_date_trans_time'].dt.day_name()
            input_feature_train_df['month'] = input_feature_train_df['trans_date_trans_time'].dt.month
                
            #clean merchant column
            input_feature_train_df['merchant'] = input_feature_train_df['merchant'].apply(lambda x : x.replace('fraud_',''))

            #Date of birth --> Age of customer
            input_feature_train_df['dob'] = pd.to_datetime(input_feature_train_df['dob'],format='mixed')
            input_feature_train_df['age'] = (input_feature_train_df['trans_date_trans_time'].dt.year - input_feature_train_df['dob'].dt.year).astype(int)

            # Calculate distance for each row using vectorized operation
            input_feature_train_df['distance_km'] = round(input_feature_train_df.apply(lambda row: great_circle((row['lat'], row['long']), (row['merch_lat'], row['merch_long'])).kilometers, axis=1), 2)

            #We will get the time between transactions for each card
            #Time=0 for every first transaction and time will be represented in hours.
            input_feature_train_df.sort_values(['cc_num', 'trans_date_trans_time'])
            input_feature_train_df['hours_diff_bet_trans']=((input_feature_train_df.groupby('cc_num')[['trans_date_trans_time']].diff())/np.timedelta64(1,'h'))
            input_feature_train_df.loc[input_feature_train_df['hours_diff_bet_trans'].isna(),'hours_diff_bet_trans'] = 0
            input_feature_train_df['hours_diff_bet_trans'] = input_feature_train_df['hours_diff_bet_trans'].astype(int)

            #Handling and extracting features from cc_num
            freq = input_feature_train_df.groupby('cc_num').size()
            input_feature_train_df['cc_freq'] = input_feature_train_df['cc_num'].apply(lambda x : freq[x])

            #Make day feature numerical
            input_feature_train_df['day'] = input_feature_train_df['trans_date_trans_time'].dt.weekday
                

            input_feature_test_df['trans_date_trans_time'] = pd.to_datetime(input_feature_test_df['trans_date_trans_time'],format='mixed')
                
            input_feature_test_df['hour'] = input_feature_test_df['trans_date_trans_time'].dt.hour
            input_feature_test_df['day'] = input_feature_test_df['trans_date_trans_time'].dt.day_name()
            input_feature_test_df['month'] = input_feature_test_df['trans_date_trans_time'].dt.month
                
            #clean merchant column
            input_feature_test_df['merchant'] = input_feature_test_df['merchant'].apply(lambda x : x.replace('fraud_',''))

            

            #Date of birth --> Age of customer
            input_feature_test_df['dob'] = pd.to_datetime(input_feature_test_df['dob'],format='mixed')
            input_feature_test_df['age'] = (input_feature_test_df['trans_date_trans_time'].dt.year - input_feature_test_df['dob'].dt.year).astype(int)

            # Calculate distance for each row using vectorized operation
            input_feature_test_df['distance_km'] = round(input_feature_test_df.apply(lambda row: great_circle((row['lat'], row['long']), (row['merch_lat'], row['merch_long'])).kilometers, axis=1), 2)

            #We will get the time between transactions for each card
            #Time=0 for every first transaction and time will be represented in hours.
            input_feature_test_df.sort_values(['cc_num', 'trans_date_trans_time'])
            input_feature_test_df['hours_diff_bet_trans']=((input_feature_test_df.groupby('cc_num')[['trans_date_trans_time']].diff())/np.timedelta64(1,'h'))
            input_feature_test_df.loc[input_feature_test_df['hours_diff_bet_trans'].isna(),'hours_diff_bet_trans'] = 0
            input_feature_test_df['hours_diff_bet_trans'] = input_feature_test_df['hours_diff_bet_trans'].astype(int)


            #Handling and extracting features from cc_num
            freq = input_feature_test_df.groupby('cc_num').size()
            input_feature_test_df['cc_freq'] = input_feature_test_df['cc_num'].apply(lambda x : freq[x])

            #Make day feature numerical
            input_feature_test_df['day'] = input_feature_test_df['trans_date_trans_time'].dt.weekday


            #drop columns
            drop_col = ['lat','long','merch_lat','merch_long', 'cc_num','trans_date_trans_time','city_pop','dob']

            input_feature_train_df = input_feature_train_df.drop(columns=drop_columns,axis=1)
            input_feature_test_df = input_feature_test_df.drop(columns=drop_columns,axis=1)
            
            ## Trnasformating using preprocessor obj
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")
            

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

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