# エラー構文
## Annihilation of the first spin contaminant
- SCF=QCを指定して計算  
計算が長くなる  

## Atomic number out of range for {汎関数名} basis
汎関数の適用範囲外の原子が含まれている場合に起こる
【対処法】  
汎関数のレベル上げる等の対策

# キーワード
# POP
分子軌道を出力する  
POP=Fullですべての軌道情報を出力  

# GaussianのMOsの読み取り型
- `Tool` > `MOsを選択`
- 表示したい軌道を選択
- `Visualize`バーを選択
- `Update`を押してしばらくすると表示される

# その他
## 構造が壊れたとき
- SCF=xqcを指定  
-> うまく行かなかった
- opt=modredundant  
-> うまく行かなかった
- opt=rfo  
-> うまく行かなかった
- 手動で構造を作る、Gaussianのクリーンアップを使わないと良いかも  
->自分で一番いいと思う初期構造を書くと良い
- 諦める。まじで無理な時がある
- rdkitの初期配座をMMFFからUFFに変換  
-> 一番いいかも!!


## 虚振動の取り扱い
振動解析を行ったときに振動数が負の値になることがある。このとき正しく振動計算ができていない。理由は不適切な構造ができていることが原因。再度構造最適化を行う必要がある。
1. 虚振動を示したファイルをGaussianで開く
1. `Results` > `Vibrations` > `Manual Displacement`のバーを-1もしくは1に設定 > `Save Structure` のボタンを押して構造を保存
1. 2で保存した構造を再び最適化
1. 3の構造ファイルを使って再度振動計算

# BDEが30以下の分子
アセトキシラジカルが壊れている可能性が高い