# photopolymerization_initiator

# 既存化合物

## OXE01
<img src="https://github.com/poclab-web/photopolymerization_initiator/blob/main/misc/OXE01.jpg" height="120px">

## OXE02
<img src="https://github.com/poclab-web/photopolymerization_initiator/blob/main/misc/OXE02.jpg" height="120px">

## OXE03
<img src="https://github.com/poclab-web/photopolymerization_initiator/blob/main/misc/OXE03.jpg" height="120px">

## OXE04
<img src="https://github.com/poclab-web/photopolymerization_initiator/blob/main/misc/OXE04.jpg" height="120px">

## OXE05
<img src="https://github.com/poclab-web/photopolymerization_initiator/blob/main/misc/OXE05.jpg" height="120px">

# Gaussianの連続計算

## 1. Gaussianのパスを通す
bash上で行った。bash_profileファイルに以下のコードを追加し、パスを通した

`export g16:$g16:/Applications/g16/g16`

一般的にパスを通したいときは以下を実行

`export 通したいパスの名前: $通したいパスの名前:パス`

bash_profileの編集がおわったら。ターミナル上でシェルの切り替えを行う。既にシェルがbashの場合は必要なし。現在のシェルはターミナルを開いてタブ（上のバー）から確認できる。

以下のコマンドを実行

`chsh -s /bin/bash`

ちなみに、他のシェルに切り替えたい場合は、

`chsh -s /bin/切り替えたいシェル名`

でいける。切り替えたのに上のタブの表示は変わらないが、これは一旦ターミナルを閉じればよし。

## 2. gjfファイルの作成
条件文に
`# "何かしらの条件" geom=connectivity`

とかくとうまくいかなかった。geom=connectivityを消すとうまくいった。つまり最終的には以下のようになる。
<sub>※geom=conectivityはファイル中に結合様式が記述されている場合に書く</sub>

<記述例> `# opt freq 6-31G(d)`

## 3. 実行の注意
出力ファイルはカレントディレクトリにできてしまう。したがって、計算の都度カレントディレクトリを変更しながら計算したほうがいい。

<計算例>

わかりやすいように変数、操作は日本語にします。

```
for i in range(計算回数):

	os.chdir(出力したいディレクトリ)

	計算対象のファイルのパス = 計算ファイルのパスを取得する操作

	!g16 $計算対象のファイルパス
```

`9/10追記`

```
!g16 $計算対象のファイルパス
```
よりも
```
import subprocess
subprocess.run('g16 計算対象のファイルパス', shell=True)
```
の方がアルゴリズムとして美しい気がする。!でshellを動かすのはipynbでは可能だがpythonスクリプトでは不可能なため

## 4.chkのfchk化
chkファイルでは計算コンピュータ以外のpcで閲覧できないので、fchkにすると便利。それを実行するのは以下のコマンド。
chkファイル名で変換する場合は、ターミナルのカレントディレクトリをファイルが存在するディレクトリに設定するように！

`$ formchk chkファイル名 or chkパス ファイル名 or パス.fchk`

## 5. 最終的な実行
Gaussianの連続計算にはgaussian/calc.pyを使うと便利。
calc.pyにあるGaussianSequentialCalculation, もしくはGaussianSequentialCalculationFromSmilesを使う。

### 5-1. GaussianSequentialCalculation
initial_gjf_path引数に最初に計算するgjfを指定する必要あり!
calc_conditions引数には計算条件を辞書型で書く
【例】
```
calc_conditions = {
	'key1': '# opt b3lyp/6-31g(d)',
	'key2': '# opt freq b3lyp/6-311++g(d)',
	・・・
	'keyn': '# opt freq b3lyp/6-311++g(2d,p)'
}
```

このgjfはあらかじめ作る必要がある。もしも作るのがめんどくさかったりしたらGaussianSequentialCalculationFromSmilesを使う

### 5-2. GaussianSequentialCalculationFromSmiles
計算したい化合物のSmilesさえわかっていれば連続計算をしてくれる
引数:SmilesにSmiles文字列を入れる。RDkitでmolオブジェクトに変換されるが、このときうまくいかないとエラーを吐くように設定。エラーがはかれるときは入力したSmilesを確認。
引数:calculation_conditionsに5-1同様の辞書型のデータを入れる