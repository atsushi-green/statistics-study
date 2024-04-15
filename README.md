<div align="center">
  <img src="https://img.shields.io/badge/rye-0.32-F26649?logo=Rye" alt="rye">
  <img src="https://img.shields.io/badge/python-3.11-F26649?logo=python" alt="python">
  <img src="https://img.shields.io/badge/scikitlearn-1.4.2-F26649?logo=scikitlearn" alt="rye">
  
</div>

# setup
```bash
$ rye sync
```

# Statistics Study

## Missing Data
欠損データの欠損メカニズムと、補完方法についてのシミュレーションを行う。欠損メカニズムに対して、各種補完方法を用いるとどのような偏りが生じるかを図で確認する。

欠損メカニズムは以下の3つを実装。
- MCAR(Missing Completely At Random): 完全にランダムに欠損が発生する
- MAR(Missing At Random): 他の変数に依存して欠損が発生する
- MNAR(Missing Not At Random): 欠損の発生が欠損値自体に依存している

補完方法として以下の2つを実装。
- 平均値で補完
- 回帰分析による補完

### 実行方法

```bash
$ rye run python scripts/run_missing_data.py
```

### 実行結果
figs/missing_fill 以下に各欠損メカニズムと補完方法の結果が保存される。


