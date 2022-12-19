from rdkit import Chem
import requests
import warnings
from IPython.display import display,HTML

class SearchFromScifinder:
    def __init__(self, smiles, warn=True):
        """SmilesからScifinderの情報を取り入れる

        Args
            smiles: Smiles文字列
            warn: 存在しない化合物のとき警告するかどうか

        Examples
            finder = SearchFromScifinder('c1cccc1')
            finder.name()
            Out: benzene
        
        Note
            json本体を持ってきたい場合は
            finder = SearchFromScifinder('c1cccc1')
            json = finder.json
        """
        self.smiles = smiles
        self.json = self.search()
        if int(self.json['count']) == 0 and warn:
            warnings.warn('json not found from: {} \n Check the Smiles you input or the Smiles not exists in Scifinder-n'.format(self.url))

    def search(self):
        mol = Chem.MolFromSmiles(self.smiles)
        inchi = Chem.MolToInchi(mol)
        self.url = 'https://commonchemistry.cas.org/api/search?q={}'.format(inchi)
        req = requests.get(self.url)
        return req.json()
    
    def name(self):
        return self.json['results'][0]['name']
    
    def cas(self):
        return self.json['results'][0]['rn']
    
    def image(self):
        """
        jupyter notebook上に画像を表示(svg形式)
        """
        return display(HTML(self.json['results'][0]['image']))
    
    def is_unknown_compound(self):
        """Scifinder-n上に存在する化合物か調査
        Returns
            存在しない -> True
            存在する -> False
        """
        result = None
        if int(self.json['count']) == 0:
            result = True
        else:
            result = False
        return result
    

