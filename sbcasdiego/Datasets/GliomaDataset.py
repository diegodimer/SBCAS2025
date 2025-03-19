import pandas as pd

from sbcasdiego.Datasets.BaseDataset import BaseDataset


class GliomaDataset(BaseDataset):

    def __init__(self, dataset=None):
        super().__init__()
        self.dataset = (
            dataset
            if dataset is not None
            else self.custom_preprocessing(
                pd.read_csv(
                    "datasets/TCGA_InfoWithGrade.csv").drop_duplicates()
            )
        )
        self.predicted_attr = "Grade"
        self.max_iter = 2000
        self.n_estimators = 20
        self.random_state = 42
        self.max_depth = 7
        self.criterion = "entropy"
        self.positive_outcome = 0
        self.protected_attr = ["Gender", "Race"]
        self.num_repetitions = 10
        self.protected_attr_mappings = {
            "Gender": {
                "Female": [0],
                "Male": [1]
            },
            "Race": {
                "Non-White": [0],
                "White": [1]
            }
        }

    def custom_preprocessing(self, df):
        def discretize_race(x):
            if x == 0:
                return 1
            else:
                return 0

        df["Race"] = df["Race"].apply(lambda x: discretize_race(x))
        return df

    def get_metrics(self, df_train, print_metrics=True):
        d = self.evaluate_metrics(
            "Gender", 1, "Age_at_diagnosis", df_train, print_metrics=print_metrics
        )
        # d.update(
        #     self.evaluate_metrics(
        #         "Education", [2,3,4,5,6], "Age_at_diagnosis", df_train, print_metrics=print_metrics
        #     )
        # )
        d.update(
            self.evaluate_metrics(
                "Race", 1, "Age_at_diagnosis", df_train, print_metrics=print_metrics
            )
        )
        return d
