import japanize_matplotlib  # noqa
import matplotlib.pyplot as plt
import numpy as np

from statistics_study.colors import ColorCode

# 正方形をサンプリングして面積を推定する回数
# この数が多いと分布が安定してヒストグラムが滑らかになる
NUM_TRIAL = 10000

# サンプルサイズ: 一回の試行で生成する正方形の数。
# - この値が小さいと不偏性の影響が大きくなる
# - この値が大きいと、中心極限定理によりヒストグラムが正規分布に近づく
SAMPLE_SIZE = 5


class SquareData:
    # 母数
    MU = 5
    SIGMA = 1
    S = MU**2

    def __init__(self, sample_size: int):
        # 正方形の1辺の長さを正規分布に従って生成
        self.side_lengthes = np.random.normal(
            loc=self.MU, scale=self.SIGMA**2, size=sample_size
        )
        self.sample_size = len(self.side_lengthes)

    def estimate_s1(self) -> float:
        # 面積を計算してから平均をとる
        return np.mean(self.side_lengthes**2)

    def estimate_s2(self) -> float:
        # 辺の長さの平均をとってから2乗する
        return np.mean(self.side_lengthes) ** 2

    def estimate_s3(self) -> float:
        # ddof=1で不偏分散
        return np.mean(self.side_lengthes) ** 2 - (
            np.var(self.side_lengthes, ddof=1) / self.sample_size
        )


class Distribution:
    BINS = 40

    def __init__(self, dist_name: str = ""):
        self.estimated_s: list[float] = []  # 推定した面積のリスト
        self.dist_name: str = dist_name
        self.mean: float = None
        self.std: float = None
        self.bias: float = None
        self.variance: float = None

    def draw_histgram(self):
        """
        面積の推定値のヒストグラムを描画。
        辺の長さが正規分布に従うため、その二乗は正規分布に従わないず、
        標準偏差が1ならば、非心カイ二乗分布に従う。
        """
        fig, ax = plt.subplots()
        ax.hist(
            self.estimated_s, bins=self.BINS, color=ColorCode.SYUIRO.value, align="mid"
        )
        ax.axvline(SquareData.S, color=ColorCode.BLUE.value, label="母数")
        y_max = 30 * NUM_TRIAL / self.BINS / SAMPLE_SIZE

        # 統計量を表示
        ax.text(
            (SquareData.S * 1.5),
            y_max / 1.5,
            f"mean: {self.mean:.2f}\nstd: {self.std:.2f}\nbias: {self.bias:.2f}\nvariance: {self.variance:.2f}",
        )

        ax.set_title(self.dist_name)
        ax.set_xlabel("推定した面積")
        ax.set_ylabel("頻度")
        ax.set_xlim(SquareData.S - SquareData.S, SquareData.S + SquareData.S)
        ax.set_ylim(0, y_max)

        fig.savefig(
            f"figs/unbiased_consistent_estimator/{self.dist_name}_estimated_s.png"
        )

    def cals_statistics(self):
        self.mean = np.mean(self.estimated_s)
        self.std = np.std(self.estimated_s)
        self.bias = np.mean(self.estimated_s) - SquareData.S
        self.variance = np.var(self.estimated_s)

        print(f"====={self.dist_name}=====")
        print(f"mean: {self.mean}")
        print(f"std: {self.std}")
        print(f"bias: {self.bias}")
        print(f"variance: {self.variance}")
        print()


# usage
if __name__ == "__main__":
    s1_dist = Distribution("面積を計算してから平均をとる")
    s2_dist = Distribution("辺の長さの平均をとってから2乗する")
    s3_dist = Distribution("バイアスを取り除く")

    for trial in range(NUM_TRIAL):
        square_data = SquareData(SAMPLE_SIZE)
        s1_dist.estimated_s.append(square_data.estimate_s1())
        s2_dist.estimated_s.append(square_data.estimate_s2())
        s3_dist.estimated_s.append(square_data.estimate_s3())

    for s_dist in [s1_dist, s2_dist, s3_dist]:
        s_dist.cals_statistics()
        s_dist.draw_histgram()
