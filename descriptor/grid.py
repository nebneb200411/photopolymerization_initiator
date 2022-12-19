import rdkit 
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

def grid_morganfingerprint(mol, radius=4, n_Bits=4096, molsPerRow=5):
    bitI_morgan = {}
    fp_morgan = AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_Bits, bitInfo=bitI_morgan)

    morgan_turples = ((mol, bit, bitI_morgan) for bit in list(bitI_morgan.keys()))
    img = Draw.DrawMorganBits(morgan_turples, molsPerRow=5,
        legends=['bit: '+str(x) for x in list(bitI_morgan.keys())])
    img.save('./result/bit_several_molecule_morgan.png',bbox_inches='tight')

def grid_morganfingerprint_with_target(mol, target, radius=4, n_Bits=4096, molsPerRow=5):
    bitI_morgan = {}
    fp_morgan = AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_Bits, bitInfo=bitI_morgan)
    morgan_turples = ((mol, bit, bitI_morgan) for bit in list(bitI_morgan.keys()) if str(bit) in target)
    img = Draw.DrawMorganBits(morgan_turples, molsPerRow=molsPerRow,
        legends=['bit: '+str(x) for x in target])
    img.save('./result/bit_several_molecule_morgan.png',bbox_inches='tight')