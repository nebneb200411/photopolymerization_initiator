import os
import warnings

class LoadCalcResultFromOutFile:
    def __init__(self, out_path):
        self.out_path = out_path
    
    def is_normally_ended(self):
        if os.path.exists(self.out_path):
            with open(self.out_path, 'r') as f:
                lines = f.readlines()
                result_line = [l for l in lines if 'JOB ENDED NORMALLY' in l]
                if result_line:
                    return True
                else:
                    return False
        
        else:
            warnings.warn('File Not Found!!')
            return None
    
    def HOMO_LUMO_gap(self):
        with open(self.out_path, 'r') as f:
            lines = f.readlines()
            target = [line for line in lines if 'HOMO LUMO ENERGIES' in line]
            target = target[0].split()
            gap = float(target[6]) - float(target[5])
            f.close()
        return gap
    
    def alpha_SOMO_LUMO_gap(self):
        with open(self.out_path, 'r') as f:
            lines = f.readlines()
            target = [line for line in lines if 'ALPHA SOMO LUMO (EV)' in line]
            target = target[0].split()
            gap = float(target[6]) - float(target[5])
            f.close()
        return gap
    
    def beta_SOMO_LUMO_gap(self):
        with open(self.out_path, 'r') as f:
            lines = f.readlines()
            target = [line for line in lines if 'BETA  SOMO LUMO (EV)' in line]
            target = target[0].split()
            gap = float(target[6]) - float(target[5])
            f.close()
        return gap
    
    def somo_level(self):
        with open(self.out_path, 'r') as f:
            lines = f.readlines()
            target = [line for line in lines if 'BETA  SOMO LUMO (EV)' in line]
            target = target[0].split()
            somo = float(target[5])
            f.close()
        return somo