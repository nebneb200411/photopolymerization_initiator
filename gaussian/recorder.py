import pandas as pd
import datetime
from .calculation_checker import check_from_logfile

class CalcualationRecorder:
    def __init__(self):
        self.df = pd.DataFrame()

    def record(self, smiles, log_path, calc_condition, detect_normal_freq):
        self.df.loc[smiles] = None
        result = check_from_logfile(log_path, detect_normal_freq=detect_normal_freq)
        self.df.at[smiles, calc_condition] = result
    
    def record_columns(self, calc_condition_dict):
        self.df.columns = calc_condition_dict.values()
    
    def save(self):
        save_date = datetime.datetime.now().strftime('%YH%mM%dD_%H-%M-%S')
        self.df.to_csv('gaussian_calc_record{}.csv'.format(save_date))
