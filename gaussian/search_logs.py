import os, sys
from smiles.change_format import gen_alt_smiles
sys.path.append('../')
import glob
from gaussian.generate_gjf import path_generator

class SearchLogsFromCalcCondition:
    def __init__(self, BASE_DIR):
        self.base_dir = BASE_DIR
    
    def search(self, smiles, calc_level, state='ground'):
        filename = path_generator(smiles, calc_level, state, extension='.log')
        log_path = os.path.join(self.base_dir, filename)
        if not os.path.exists(log_path):
            return None
        return log_path

class SearchLogsFromKeyword(SearchLogsFromCalcCondition):
    """ Gaussianで計算したLogファイルの中からSmiles文字列に一致する化合物を計算
    対象は、以下のような形式のlogファイル
    「 変換したSmiles文字列_計算条件.log」
    keyword -> 探したい化合物のファイル名のキーワード。計算条件(freq, opt)などを入れると良い
    BASE_DIR -> 探したいLogファイルが入ったDirectory
    """
    def __init__(self, keyword, BASE_DIR):
        super().__init__(BASE_DIR)
        self.base_dir = BASE_DIR
        self.keyword = keyword
        self.all_logs = self.get_all_calculated_alt_smiles()
    
    def get_all_calculated_alt_smiles(self):
        all_logs = glob.glob(os.path.join(self.base_dir, '*{}*.log'.format(self.keyword)))
        all_logs = [os.path.basename(x) for x in all_logs]
        return all_logs
    
    def search(self, smiles):
        """
        同じキーワードの中から複数のファイルが見つかった場合、更新日が最新のファイルを取得
        """
        alt_smiles = gen_alt_smiles(smiles)
        detected_logs = [x for x in self.all_logs if x.split('_')[0] == alt_smiles]
        if len(detected_logs) == 1:
            detected_logs = detected_logs[0]
        elif not detected_logs: # if there is no log file's in the directory
            print('file not found!!')
            print(smiles)
            detected_logs = 'ERROR.log'
        else: # use additional keyword if there is several logs
            import warnings
            warnings.warn('We detected several log files from input keyword.\n detected log files:{}'.format(detected_logs))
            time_cache = []
            # get file's updated time
            for log in detected_logs:
                log_path = os.path.join(self.base_dir, log)
                updated_at = os.path.getmtime(log_path)
                time_cache.append(updated_at)
            # detect updated file
            max_time = max(time_cache)
            max_index = time_cache.index(max_time)
            detected_logs = detected_logs[max_index]
        return detected_logs

class SearchLogFromKeywords(SearchLogsFromKeyword):
    def __init__(self, keywords:list, base_dir:str):
        """ キーワードをリストで入れて取り出す

        Args
            keywords: キーワードのリスト
            base_dir: 結果を保存するディレクトリ
        
        Note
            Smilesファイル名_計算条件.logの形式を想定
            複数のlogファイルが見つかった場合、変更日が最新のものを取得
        """
        self.keywords = keywords
        self.base_dir = base_dir
        assert type(self.keywords) == list, 'Keywords must be list type!!'
        assert os.path.exists(self.base_dir), 'Directory Not Exists!!'
        self.all_logs = glob.glob(os.path.join(self.base_dir, '*.log'))
        self.all_logs = [os.path.basename(x) for x in self.all_logs]
    
    def search(self, smiles):
        """ Smiles文字列_計算条件等.logから目的のlogファイルを抽出

        Args
            smiles: Smiles文字列を入力
            
        """
        alt_smiles = gen_alt_smiles(smiles)
        keywords = self.keywords.copy()
        extracted = self.all_logs
        extracted = [x for x in extracted if x.split('_')[0] == alt_smiles]
        for keyword in keywords:
            extracted = [log for log in extracted if keyword in log]
            if len(extracted) == 0:
                return None
        if len(extracted) > 1:
            log_file = self.get_latest_modified_file(extracted)
        else:
            log_file = extracted[0]
        
        log_file = os.path.join(self.base_dir, log_file)
        return log_file
    
    def get_latest_modified_file(self, files):
        time_cache = []
        # get file's updated time
        for f in files:
            path = os.path.join(self.base_dir, f)
            updated_at = os.path.getmtime(path)
            time_cache.append(updated_at)
        # detect updated file
        max_time = max(time_cache)
        max_index = time_cache.index(max_time)
        latest_mod_file = files[max_index]
        latest_mod_file = os.path.basename(latest_mod_file)
        return latest_mod_file



