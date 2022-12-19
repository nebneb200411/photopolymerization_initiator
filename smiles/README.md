# smiles
Smiles文字列に対して行う操作を網羅

## 1.change_format.py
smiles文字列をファイル名に変換するときに使う
/や\が入っているとファイルをうまく保存できないため

## 2.fragment.py
分子の結合を切るときに使う
現在は脱炭酸のみ

## 3.grid_mol.py
分子の描写
単一の分子から複数分子まで描ける
grid_imgは計算した化合物をリストで入力。入力する分子はSmilesもしくはmolオブジェクト。

## 4.ketone_to_oximester.py
ケトン→オキシムエステルに変換する
【メモ】ディレクトリ名はcompounds_conversion.pyにした方がいいかも

## 5.substructure
部分構造探索