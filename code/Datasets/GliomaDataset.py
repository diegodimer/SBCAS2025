import pandas as pd
from Datasets.BaseDataset import BaseDataset


class GliomaDataset(BaseDataset):

    def __init__(self, dataset=None):
        super().__init__()
        self.dataset = (
            dataset
            if dataset is not None
            else self.custom_preprocessing(
                pd.read_csv("datasets/TCGA_GBM_LGG_Mutations_all.csv")
            )
        )
        self.predicted_attr = "Grade"
        self.max_iter = 2000
        self.n_estimators = 20
        self.random_state = 0
        self.max_depth = 7
        self.criterion = "entropy"
        self.positive_outcome = 0
        self.protected_attr = ["Gender", "Race"]
        self.num_repetitions = 10
        self.protected_attr_mappings = {
            "Gender": {
                "Female": 0, 
                "Male": 1
                },
            "Race": {
                "White": 1, 
                "Non-White": 0
                }
        }

    def custom_preprocessing(self, df):
        def discretize_sex(x):
            if x == "Female":
                return 0
            elif x == "Male":
                return 1
            else:
                raise

        def discretize_race(x):
            if x == "white":
                return 1
            else:
                return 0

        df["Sex"] = df["Sex"].apply(lambda x: discretize_sex(x))
        df["Race"] = df["Race"].apply(lambda x: discretize_race(x))

        return df

    def get_metrics(self, df_train):
        raise NotImplementedError("Method not implemented")