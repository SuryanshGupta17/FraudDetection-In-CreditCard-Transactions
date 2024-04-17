import os 
import sys 
import pandas as pd
import mysql.connector
from sqlalchemy import create_engine
import numpy as np
import pickle 
from geopy.distance import great_circle
from src.exception import CustomException
from sqlalchemy import create_engine
from datetime import time
from sklearn.metrics import accuracy_score
from src.logger import logging

def save_function(file_path, obj): 
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok= True)
    with open (file_path, "wb") as file_obj: 
        pickle.dump(obj, file_obj)

def model_performance(X_train, y_train, X_test, y_test, models): 
    try: 
        report = {}
        
        def evaluate_model(model, X_train, y_train, X_test, y_test):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            test_model_score = accuracy_score(y_test, y_pred)
            
            return test_model_score

        for name, model in models.items():
            test_model_score = evaluate_model(model, X_train, y_train, X_test, y_test)
            report[name] = test_model_score

        
        return report

    except Exception as e: 
        raise CustomException(e,sys)

# Function to load a particular object 
def load_obj(file_path):
    try: 
        with open(file_path, 'rb') as file_obj: 
            return pickle.load(file_obj)
    except Exception as e: 
        logging.info("Error in load_object fuction in utils")
        raise CustomException(e,sys)


def feature_engg(test_df):
    test_df['trans_date_trans_time'] = pd.to_datetime(test_df['trans_date_trans_time'],format='mixed')

    test_df['hour'] = test_df['trans_date_trans_time'].dt.hour
    test_df['day'] = test_df['trans_date_trans_time'].dt.day_name()
    test_df['month'] = test_df['trans_date_trans_time'].dt.month
            
    test_df['merchant'] = test_df['merchant'].apply(lambda x : x.replace('fraud_',''))

    test_df['dob'] = pd.to_datetime(test_df['dob'],format='mixed')
    test_df['age'] = (test_df['trans_date_trans_time'].dt.year - test_df['dob'].dt.year)

    test_df['distance_km'] = round(test_df.apply(lambda row: great_circle((row['lat'], row['long']), (row['merch_lat'], row['merch_long'])).kilometers, axis=1), 2)

    test_df.sort_values(['cc_num', 'trans_date_trans_time'],inplace=True)
    test_df['hours_diff_bet_trans']=((test_df.groupby('cc_num')[['trans_date_trans_time']].diff())/np.timedelta64(1,'h'))


    freq = test_df.groupby('cc_num').size()
    test_df['cc_freq'] = test_df['cc_num'].apply(lambda x : freq[x])

    def class_det(x):
                for idx,val in enumerate(list(range(800,5000,800))):
                    if x < val:
                        return idx+1
                    
    test_df['cc_freq_class'] = test_df['cc_freq'].apply(class_det)

    test_df['day'] = test_df['trans_date_trans_time'].dt.weekday

    return test_df
            
def import_data_from_mysql():
    connection = connect_to_mysql()
    
    query = "SELECT * FROM fraudprojectdata"
    
    filename = "data/Train.csv"
    export_data(connection, query, filename)
    return filename
    
def connect_to_mysql():
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="Shanky17",
            database="fraudproject",
            use_pure=True
        )
        return connection
    except Exception as e: 
        raise CustomException(e,sys)

def export_data(connection, query, filename):
    try:
        cursor = connection.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w') as file:
            for row in rows:
                file.write(','.join(map(str, row)) + '\n')
        print("Data exported and saved to file:", filename)
    except mysql.connector.Error as error:
        print("Failed to export data from MySQL:", error)

