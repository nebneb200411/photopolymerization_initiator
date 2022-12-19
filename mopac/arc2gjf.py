import os, sys
sys.path.append('../')
from files.extension import ExtensionChanger
from smiles.change_format import gen_alt_smiles
from gaussian.generate_gjf import change_calc_level_format
from dotenv import load_dotenv
load_dotenv()

def arc_to_gjf(arc_path, base_dir, gaussian_calc_condition, charge_and_mult ,**link0_command):
    """ 
    【仕様】
    MOPACで出力したarcファイルをgjfに変換する
    【引数の説明】
    arc_pathにMOPACで計算して出力したarcファイルのパスを指定
    base_dirにgjfを出力したいDirectoryを指定
    gaussian_calc_conditionにGaussian形式で計算条件をかく
    link0_commandにgaussianの計算オプションを指定
    """
    assert os.path.exists(arc_path), 'dat file not exists!!'
    with open(arc_path, 'r') as f:
        lines = f.readlines()
        title_index = [idx for idx, x in enumerate(lines) if os.environ['MOPAC_TITLE'] in x][-1]
        xyz = lines[title_index+1:-1]
        xyz = [x.split() for x in xyz]
        # ['atom', x_position, charge, y_position, charge, z_position]
        f.close()
    
    ext_changer = ExtensionChanger(arc_path)
    gjf_path = ext_changer.replacer('.gjf')
    gjf_path_ = os.path.basename(gjf_path)
    alt_smiles = gjf_path_.split('_')[0]
    chk_path = '{}_{}.chk'.format(alt_smiles, change_calc_level_format(gaussian_calc_condition))
    gjf_path = os.path.join(base_dir, '{}_{}.gjf'.format(alt_smiles, change_calc_level_format(gaussian_calc_condition)))

    with open(gjf_path, 'w') as f:
        for k, v in link0_command.items():
            print('%{}={}'.format(k, v), file=f)
        print('%chk={}'.format(chk_path), file=f)
        print(gaussian_calc_condition, file=f)
        print('', file=f)
        print('Good Luck!!', file=f)
        print('', file=f)
        print(charge_and_mult, file=f)
        for p in xyz:
            print(' {}  {} {} {}'.format(p[0], p[1], p[3], p[5]), file=f)
        print(' ', file=f)
        f.close()
    
    return gjf_path
