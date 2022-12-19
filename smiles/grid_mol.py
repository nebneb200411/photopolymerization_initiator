from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import os, sys
import datetime


def smiles_to_img(smiles, save_path):
    mol = Chem.MolFromSmiles(smiles)
    img = Draw.MolToImage(mol)
    img.save(save_path)

def grid_img(objects, row, save_path, subImgSize=(200, 200), use_labels=True):
    assert type(objects) == list, 'Input objects you want to grid in list type'
    if type(objects[0]) == str:
        smiles = objects
        mols = [Chem.MolFromSmiles(x) for x in objects]
    elif type(objects[0]) == Chem.rdchem.Mol:
        smiles = [Chem.MolToSmiles(x) for x in objects]
        mols = objects
    else:
        sys.stderr.write('object in the list must be mol-object or smiles')
    img = Draw.MolsToGridImage(mols, molsPerRow=row, subImgSize=subImgSize, legends=smiles)
    img.save(save_path)

def grid_mols(objects, row, subImgSize=(200, 200)):
    assert type(objects) == list, 'Input objects you want to grid in list type'
    if type(objects[0]) == str:
        smiles = objects
        mols = [Chem.MolFromSmiles(x) for x in objects]
    elif type(objects[0]) == Chem.rdchem.Mol:
        smiles = [Chem.MolToSmiles(x) for x in objects]
        mols = objects
    else:
        sys.stderr.write('object in the list must be mol-object or smiles')
    img = Draw.MolsToGridImage(mols, molsPerRow=row, subImgSize=subImgSize, legends=smiles)
    return img

def grid_morganfingerprint(df, grid_bits, radius, nBits):
    """特定のビットを描画する
    """
    if not os.path.exists('./result/MorganFingerprint'):
        os.makedirs('./result/MorganFingerprint', exist_ok=True)
    
    time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    directory = './result/MorganFingerprint/{}'.format(time)
    os.makedirs(directory)

    for bit in grid_bits:
        target = df[df[bit] == 1]
        if not len(target) == 0:
            obj = target.iloc[0]['Smiles']
            mol = Chem.MolFromSmiles(obj)
            bitI_morgan = {}
            fp_morgan = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits, bitInfo=bitI_morgan)
            for i in range(1, len(target)):
                try:
                    img = Draw.DrawMorganBit(mol, int(bit), bitI_morgan)
                    img.save(os.path.join(directory, '{}.png'.format(bit)))
                except:
                    obj = target.iloc[i]['Smiles']