from missing_data.missing_fill import XYData


def main():
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


if __name__ == "__main__":
    main()
