from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import math
from scipy.stats import skew
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report


class NaiveBaselineModel:
    def __init__(self, df, PLOT_PATH, LOG_PATH):
        self.df = df
        self.PLOT_PATH = PLOT_PATH
        self.file_name = LOG_PATH + "baseline_model"

    def get_skew(self, x, pred="current-energy-efficiency"):
        """
        Distribution of postcodes will tell us if we should aggregate by median or mean

        Input:
        x(str): Name of column for variable to detect skew
        """
        var_skew = []

        for var, agg in self.df.groupby(x, pred):
            var_skew.append(skew(agg[pred]))

        plt.hist(var_skew)
        plt.show()
        plt.savefig(self.PLOT_PATH + f"{x}_skew")

    def train_test_split(self, pred):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.df, self.df[pred], test_size=0.3, random_state=123
        )
        self.X_train[pred] = self.y_train

    def naive_model(self, group_var):
        """
        Computes the naive model performance for predicting energy efficiency.

        Input:
        group_var(str): Variable to group on
        """
        pred = "current-energy-efficiency"
        self.train_test_split(pred)

        train = self.X_train[[group_var, pred]]
        grouped_train = train.groupby(group_var).mean()

        mapping = dict(zip(grouped_train.index, grouped_train[pred]))
        y_pred = self.X_test[group_var].map(mapping)
        y_pred[np.isfinite(y_pred) == False] = grouped_train.pred.mean()

        with open(f"{self.file_name}.txt", "a") as f:
            print(f"Results for naive model on {group_var}", file=f)
            print(f"Testing r2: {r2_score(self.y_test, y_pred)}", file=f)
            print(f"Testing MSE: {mean_squared_error(self.y_test, y_pred)}", file=f)
            print(
                f"Testing Mean Error: {math.sqrt(mean_squared_error(self.y_test, y_pred))}",
                file=f,
            )

    def naive_classifier(self, group_var):
        """
        Computes the naive model performance for predicting energy rating.

        Input:
        group_var(str): Variable to group on
        """
        pred = "current-energy-rating"
        self.train_test_split(pred)

        grouped_train = (
            self.X_train[[group_var, pred]]
            .groupby(group_var)
            .agg(lambda x: pd.Series.mode(x)[0])
        )
        mapping = dict(zip(grouped_train.index, grouped_train["y"]))
        y_pred = self.X_test[group_var].map(mapping)
        y_pred = y_pred.replace(np.nan, grouped_train.pred.mode()[0])

        with open(f"{self.file_name}.txt", "a") as f:
            print(f"Results for naive classification model on {group_var}", file=f)
            print(classification_report(self.y_test, y_pred), file=f)


def main(df, PLOT_PATH, LOG_PATH):
    model = NaiveBaselineModel(df, PLOT_PATH, LOG_PATH)

    model.get_skew("postcode")
    model.get_skew("local-authority")

    model.naive_model("postcode")
    model.naive_model("local-authority")

    model.naive_classifier("postcode")
    model.naive_classifier("local-authority")


if __name__ == "__main__":
    main()
