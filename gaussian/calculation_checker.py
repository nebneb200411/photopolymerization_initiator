import os, sys
sys.path.append('../')
import glob
from .load_calc_result import LoadCalcResultFromLogFile
from smiles.change_format import gen_alt_smiles
try:
    from tqdm.notebook import tqdm
except:
    from tqdm import tqdm


class AllChecker:
    """
    【仕様】
    Gaussianの計算がうまく行ったか判定したいときに使う
    logファイルから判定
    【出力】
    リスト形式
    ['エラーが起きたパス', '原因（ファイルなし or 計算に失敗 or 虚振動）']
    エラーが起きた計算の内容 <- ファイルが見つからなかったか、計算がうまく行ってないかを判定
    【注意点】
    GaussianSequentialCalculationで計算したファイルのみの適用を想定
    """
    def __init__(self, base_dir, condition_keywords=['opt', 'freq', 'excited']):
        self.base_dir = base_dir
        self.all_logs = glob.glob(os.path.join(base_dir, '*.log'))
        self.condition_keywords = condition_keywords
        self.error_logger = []
    
    def cheker(self, detect_imaginary_freq=True, save_result=False):
        for log in tqdm(self.all_logs):
            loader = LoadCalcResultFromLogFile(log)
            base_name = os.path.basename(log)
            if not loader.is_ended_nomally():
                cause = 'file not found' if not os.path.exists(log) else 'Calculation Failed'

                self.error_logger.append([base_name, cause])

            if loader.is_ended_nomally() and 'freq' in log and detect_imaginary_freq and not loader.is_normal_frequencies():
                self.error_logger.append([base_name, 'Imaginary Frequencies'])
        if save_result:
            self.save_as_pickle(self.error_logger)
        return self.error_logger
    
    def check_from_smiles(self, smiles, detect_imaginary_freq=False):
        assert type(smiles) == list, 'you have to input smiles as list type'
        for s in smiles:
            logs = glob.glob(os.path.join(self.base_dir, gen_alt_smiles(s) + '*.log'))
            for log in logs:
                cause = 'file not found' if not os.path.exists(log) else 'Calculation Failed'
                loader = LoadCalcResultFromLogFile(log)
                base_name = os.path.basename(log)
                error_condition = [x for x in self.condition_keywords if x in base_name]
                if len(error_condition) == 1:
                    error_condition = error_condition[0]
                else:
                    error_condition = 'misc'
                if error_condition == 'freq' and detect_imaginary_freq and not loader.is_normal_frequencies():
                    self.error_logger.setdefault(base_name, ('freq', 'imaginary frequencies'))
                else:
                    self.error_logger.setdefault(base_name, (error_condition, cause))
        
        return self.error_logger
    
    def save_as_pickle(self, logger):
        import pickle
        with open(os.path.join(self.base_dir, 'error_logger_TypeDict.pickle'), 'wb') as f:
            pickle.dump(logger, f)
        print('Saving Completed!!')

def check_from_logfile(log_path, detect_normal_freq=False):
    if log_path and os.path.exists(log_path):
        loader = LoadCalcResultFromLogFile(log_path)
        if loader.is_ended_normally():
            if detect_normal_freq:
                if loader.is_normal_frequencies():
                    return 'Success'
                else:
                    return 'Imaginary Frequencies!!'
            else:
                return 'Success'
        else:
            return 'Calculation Failed!!'
    else:
        return 'File Not Found!!'