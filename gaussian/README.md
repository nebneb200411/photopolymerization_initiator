# gaussian

## 1.calc.py
計算を実行する
### 1-1.GaussianSequentialCalculation
予めGJFファイルが作成されている場合、連続計算可能
【注意】
計算条件は必ずdict型で
10/8改定:
計算のプロパティは変数として宣言する必要がある
例えば、並列計算を行いたいときは
Nprocshared='8'など
引数名がオプション名、入力値がオプションの値になる
### 1-2.GaussianSequentialCalculation
Smilesと計算条件を辞書型で渡せば連続計算してくれる。こっちの方が実用的

## 2.generate_gjf.py
gjfファイルの作成
### 2-1.GJFGenerator
smilesを入力すればGJFファイルを作成してくれる
ほとんどGaussianSequentialCalculationのためのファイルだが、個別にGJFファイルを作りたいときにも使える
【注意】
Gaussian上でGJFファイルに結合を記入していないと思っている分子と違う分子ができる。大まかな構造は変わらないが、結合が変になっていたりすることがある。このまま計算を進めると計算結果がおかしくなることがああるので`connectivity=True`を推奨

## 3.load_calc_result.py
logファイルの解析用
### 3-1.LoadCalcResultFromLogFile
logファイルを読み込んで中身を解析してくれる
詳細は中身を見ると良い
### 3-2.integrate_spectra_from_df
dataframeを入力すればスペクトルを積分してくれる
【注意】
dataframeのカラムにおいて波長をwavelength, 吸光係数をepsilonと指定しないといけないので使い勝手が悪い