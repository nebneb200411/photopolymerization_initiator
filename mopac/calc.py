from .generate_dats import DatsGenerator
from .run import run_mopac
try:
    from tqdm.notebook import tqdm
except:
    from tqdm import tqdm

class MOPACSequential:
    """smilesにsmiles文字列のリストを入れて連続計算
    """
    def __init__(self, smiles, base_dir):
        self.smiles = smiles
        assert type(self.smiles) == list, 'You have to input smiles as list type!!'
        self.base_dir = base_dir
    
    def run(self, CALC_CONDITION):
        for s in tqdm(self.smiles):
            dats_path = DatsGenerator(smiles=s, base_dir=self.base_dir, calc_condition=CALC_CONDITION).write_dat()
            run_mopac(dats_path)

def calc_from_smiles(smiles, base_dir, calc_condition, mode="UFF"):
    dats_path = DatsGenerator(smiles=smiles, base_dir=base_dir, calc_condition=calc_condition).write_dat()
    run_mopac(dats_path)
    return dats_path