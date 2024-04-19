from enum import Enum
from typing import Optional

import japanize_matplotlib  # noqa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from statistics_study.colors import ColorCode


class MissingMechanism(Enum):
    NONE = "欠損なし"
    MCAR = "MCAR"
    MAR = "MAR"
    MNAR = "MNAR"


class FillingMethod(Enum):
    NONE = "補完なし"
    MEAN = "平均補完"
    REGRESSION = "回帰補完"


class XYData:
    """
    xとyには正の相関があることを想定した1対のデータを保持するクラス
    yのデータには欠損が生じる可能性があり、そのメカニズムによって以下の3つの場合に分けて欠損を発生させる
    - MCAR(Missing Completely At Random): 完全にランダムに欠損が発生する
    - MAR(Missing At Random): 他の変数に依存して欠損が発生する
    - MNAR(Missing Not At Random): 欠損の発生が欠損値自体に依存している

    また、欠損値の補完方法として以下の2つを実装
    - 平均値で補完
    - 回帰分析による補完

    具体例としては、
    x: 中間試験の点数
    y: 期末試験の点数
    を想定し、中間試験の点数が高い人ほど期末試験の点数も高いという正の相関があるとする。
    また、中間試験の点数が

    """

    NUM_DATA = 3000
    MISSING_RATE = 0.2

    def __init__(
        self,
        x: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        missing_mechanism: Optional[MissingMechanism] = MissingMechanism.NONE,
        filling_method: Optional[FillingMethod] = FillingMethod.NONE,
    ):
        if x is None or y is None:
            self.x, self.y = self.generate_data()
        else:
            self.x = x
            self.y = y

        assert (
            len(self.x) == len(self.y) == self.NUM_DATA
        ), f"len(x)={len(self.x)} != len(y)={len(self.y)}"

        self.missing_mechanism: MissingMechanism = missing_mechanism
        self.filling_method: FillingMethod = filling_method

    @classmethod
    def generate_data(cls) -> tuple[np.ndarray, np.ndarray]:
        """正の相関を持つx-y組のデータを生成

        Returns:
            tuple[np.ndarray, np.ndarray]: 正の相関を持つx-yデータ
        """
        x = np.random.rand(cls.NUM_DATA)
        y = x + np.random.normal(scale=0.2, size=cls.NUM_DATA)
        return x, y

    def constructor(self, **kwargs) -> "XYData":
        # 元のインスタンスの属性をコピー
        attrs = self.__dict__.copy()
        # 新しい値で属性を更新
        attrs.update(kwargs)
        # 新しいインスタンスを生成
        new_instance = self.__class__(**attrs)
        return new_instance

    def missing_with_MCAR(self) -> "XYData":
        x = self.x.copy()
        y = self.y.copy()
        num_missing = int(len(self.x) * self.MISSING_RATE)  # 欠損させる件数
        all_indices = np.arange(len(self.x))
        # 欠損させるインデックス
        chosen_indices = np.random.choice(all_indices, num_missing, replace=False)
        y[chosen_indices] = np.nan
        return self.constructor(x=x, y=y, missing_mechanism=MissingMechanism.MCAR)

    def missing_with_MAR(self) -> "XYData":
        x = self.x.copy()
        y = self.y.copy()
        num_missing = int(len(x) * self.MISSING_RATE)  # 欠損させる件数
        # xが上位missing_rate%のデータを欠損させる
        x_sorted_indices = np.argsort(x)
        x_missing_indices = x_sorted_indices[-num_missing:]
        # yが上位missing_rate%のデータを欠損させる
        y[x_missing_indices] = np.nan
        return self.constructor(x=x, y=y, missing_mechanism=MissingMechanism.MAR)

    def missing_with_MNAR(self) -> "XYData":
        x = self.x.copy()
        y = self.y.copy()
        num_missing = int(len(x) * self.MISSING_RATE)
        # yが下位missing_rateのデータを欠損させる
        y_sorted_indices = np.argsort(y)
        y_missing_indices = y_sorted_indices[:num_missing]
        y[y_missing_indices] = np.nan
        return self.constructor(x=x, y=y, missing_mechanism=MissingMechanism.MNAR)

    def fillna_with_mean(self) -> "XYData":
        x = self.x.copy()
        y = self.y.copy()
        nan_indices = np.isnan(y)
        mean_y = np.mean(y[~nan_indices])
        y[nan_indices] = mean_y

        return self.constructor(x=x, y=y, filling_method=FillingMethod.MEAN)

    def fillna_with_regression(self) -> "XYData":
        x = self.x.copy()
        y = self.y.copy()
        nan_indices = np.isnan(y)
        train_x = x[~nan_indices].reshape([-1, 1])
        train_y = y[~nan_indices].reshape([-1, 1])

        # 回帰分析
        lin_reg = LinearRegression()
        lin_reg.fit(train_x, train_y)
        y_predict = lin_reg.predict(x.reshape([-1, 1]))
        y[nan_indices] = y_predict[nan_indices].flatten()

        return self.constructor(x=x, y=y, filling_method=FillingMethod.REGRESSION)

    def fit_plot(self):
        # 欠損を除く
        df = pd.DataFrame({"x": self.x, "y": self.y}).dropna()
        x = df["x"].values
        y = df["y"].values

        # 統計量の計算
        corr = np.corrcoef(x, y)[0, 1]
        y_mean = np.mean(y)
        y_std = np.std(y)

        # 回帰分析
        x = np.array([x]).reshape([-1, 1])
        y = np.array([y]).reshape([-1, 1])
        lin_reg = LinearRegression()
        lin_reg.fit(x, y)
        # 回帰直線を計算
        x_new = np.array([[0], [1]])
        y_predict = lin_reg.predict(x_new)

        # データと回帰直線をプロット
        fig, ax = plt.subplots()
        ax.scatter(x, y, s=0.5, color=ColorCode.BLUE.value)
        ax.plot(x_new, y_predict, color=ColorCode.SYUIRO.value, label="回帰直線")

        # グラフの見た目調整
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        title = f"欠損メカニズム: {self.missing_mechanism.value} 補完方法: {self.filling_method.value}  データ件数: {len(df)}\n"
        title += f"coef: {lin_reg.coef_[0][0]:.2f}  intercept: {lin_reg.intercept_[0]:.2f}, corr: {corr:.2f}, mean: {y_mean:.2f}, std: {y_std:.2f}"
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")

        # 図の保存
        savepath = f"figs/missing_fill/{self.filling_method.value}_{self.missing_mechanism.value}.png"
        print(f"[SAVE FIG] {savepath}")
        fig.savefig(savepath, bbox_inches="tight", pad_inches=0.1)

        # 図のクリア
        plt.clf()
        plt.close()
        return


if __name__ == "__main__":
    xy_data = XYData()
    xy_data.fit_plot()

    mcar_data = xy_data.missing_with_MCAR()
    mar_data = xy_data.missing_with_MAR()
    mnar_data = xy_data.missing_with_MNAR()

    for missing_data in [xy_data, mcar_data, mar_data, mnar_data]:
        # 補完なし
        missing_data.fit_plot()

        # 平均値で補完
        xydata_filled_mean = missing_data.fillna_with_mean()
        xydata_filled_mean.fit_plot()

        # 回帰分析による補完
        xydata_filled_regression = missing_data.fillna_with_regression()
        xydata_filled_regression.fit_plot()
