import sys 
import os 
from src.exception import CustomException 
from src.logger import logging 
from src.utils import load_obj
import pandas as pd

class PredictPipeline: 
    def __init__(self) -> None:
        pass

    def predict(self, features): 
        try: 
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join("artifacts", "model.pkl")

            preprocessor = load_obj(preprocessor_path)
            model = load_obj(model_path)

            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred
        except Exception as e: 
            logging.info("Error occured in predict function in prediction_pipeline location")
            raise CustomException(e,sys)
        
class CustomData: 
        def __init__(self, trans_date_trans_time:object,
                     dob:object, 
                     lat:float, 
                     long:float, 
                     merch_lat:float, 
                     merch_long:float, 
                     cc_num:int,
                     amt:float, 
                     zip:int, 
                     gender:object,
                     merchant:object, 
                     category:object, 
                     city:object, 
                     job:object): 
             self.trans_date_trans_time = trans_date_trans_time
             self.dob = dob
             self.lat = lat
             self.long = long 
             self.merch_lat = merch_lat
             self.merch_long = merch_long
             self.cc_num = cc_num
             self.amt = amt
             self.zip = zip 
             self.gender = gender
             self.merchant = merchant
             self.category = category
             self.city = city
             self.job = job
        
        def get_data_as_dataframe(self): 
             try: 
                  custom_data_input_dict = {
                       'trans_date_trans_time': [self.trans_date_trans_time], 
                       'dob': [self.dob], 
                       'lat': [self.lat],
                       'long':[self.long],
                       'merch_lat':[self.merch_lat], 
                       'merch_long':[self.merch_long], 
                       'cc_num':[self.cc_num], 
                       'amt':[self.amt], 
                       'zip': [self.zip], 
                       'gender': [self.gender],
                       'merchant': [self.merchant],
                       'category': [self.category],
                        'city' : [self.city],
                        'job' : [self.job]
                  }
                  df = pd.DataFrame(custom_data_input_dict)
                  logging.info("Dataframe created")
                  return df
             except Exception as e:
                  logging.info("Error occured in get_data_as_dataframe function in prediction_pipeline")
                  raise CustomException(e,sys) 
             
             
        