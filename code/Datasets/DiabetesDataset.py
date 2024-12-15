import pandas as pd
from Datasets.BaseDataset import BaseDataset


class DiabetesDataset(BaseDataset):

    def __init__(self, dataset=None):
        super().__init__()
        self.dataset = (
            dataset
            if dataset is not None
            else pd.read_csv("datasets/diabetes_binary_health_indicators_BRFSS2015.csv").drop_duplicates()
        )
        self.predicted_attr = "Diabetes_binary"
        self.max_iter = 2000
        self.n_estimators = 20
        self.random_state = 0
        self.max_depth = 7
        self.criterion = "entropy"
        self.positive_outcome = 0
        self.protected_attr = ["Sex", "Age", "Education", "Income"]
        self.num_repetitions = 10
        self.protected_attr_mappings = {
            "Sex": {
                "Female": 0, 
                "Male": 1},
            "Age": {
                "Age 18 - 24": 1, 
                "Age 25 to 29": 2, 
                "Age 30 to 34": 3, 
                "Age 35 to 39": 4, 
                "Age 40 to 44": 5, 
                "Age 45 to 49": 6, 
                "Age 50 to 54": 7, 
                "Age 55 to 59": 8, 
                "Age 60 to 64": 9, 
                "Age 65 to 69": 10, 
                "Age 70 to 74": 11, 
                "Age 75 to 79": 12, 
                "Age 80 or older": 13
                },
            "Education": {
                "Never attended school or only kindergarten": 1, 
                "Grades 1 - 8 (Elementary)": 2, 
                "Grades 9 - 11 (Some high school)": 3, 
                "Grade 12 or GED (High school graduate)": 4, 
                "College 1 year to 3 years (Some college or technical school)": 5, 
                "College 4 years or more (College graduate)": 6
            },
            "Income": {
                "Less than $10,000": 1,
                "Less than $15,000 ($10,000 to less than $15,000)": 2,
                "Less than $20,000 ($15,000 to less than $20,000)": 3,
                "Less than $25,000 ($20,000 to less than $25,000)": 4,
                "Less than $35,000 ($25,000 to less than $35,000)": 5,
                "Less than $50,000 ($35,000 to less than $50,000)": 6,
                "Less than $75,000 ($50,000 to less than $75,000)": 7,
                "$75,000 or more": 8
            }

        }

    def get_metrics(self, df_train):
        raise NotImplementedError("Method not implemented")
