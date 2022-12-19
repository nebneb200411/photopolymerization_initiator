# molオブジェクト
2つの分子が同じ分子でもSmilesがちがうと、違うMolオブジェクトとして判定される
2つの分子が同じかどうかを判定したいときは
molオブジェクトに変換 -> Smilesに変換
して同じかどうか判定する

# Smilesをデータフレームに保存するとき
molオブジェクトに変換して再度Smilesに変換してから保存したほうがいい
重複が消えるため

# 分子の結合を切るときのコツ
FragmentOnBondsで結合を切ってからGetMolFragsで切った分子をmolオブジェクトに変換。ラジカル等には変換できないので注意
```
mol = 切りたい化合物のMolオブジェクト
cut_atom_index1 = 切りたい構造のインデックス1つめ
cut_atom_index2 = 切りたい構造のインデックス2つめ
frags = Chem.FragmentOnBonds(mol, [cut_atom_index1, cut_atom_index2], addDummies=False) # ここでaddDummies=Falseにするとダミーラベルが含まれなくなる。どこで結合を切ったかを記録する必要がないときはfalseに。
frags = Chem.GetMolFrags(frags, asMols=True) # ここで作ったフラグメントをMolオブジェクトのリストにする
```

# 記述子関連
## Numrotatablbond
剛直すぎると広がりが少ない

## NumSatulated

## TPSA
算出方法: https://future-chem.com/tpsa/ 
参考: https://zenn.dev/poclabweb/books/chemoinfomatics_beginner/viewer/lesson04_02_1_rdkit

## Fracrtion SP3
rdkit.Chem.rdMolDescriptors.CalcFractionCSP3