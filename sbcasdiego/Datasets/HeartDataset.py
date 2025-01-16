import pandas as pd
from sbcasdiego.Datasets.BaseDataset import BaseDataset


class HeartDataset(BaseDataset):
    """
    HeartDataset class for handling heart disease dataset.
    This class inherits from BaseDataset and is used to manage and preprocess the heart disease dataset.
    It provides functionalities to initialize the dataset, set various attributes, and evaluate metrics.

    Methods
    -------
    __init__(self, dataset=None):
        Initialize the HeartDataset class with the given dataset or load from a CSV file.
    get_metrics(self, df_train):
        Evaluate and return metrics for the dataset based on protected attributes.

    """

    def __init__(self, dataset=None):
        """
        Initialize the HeartDataset class.

        Args:
        ----
        dataset (pd.DataFrame, optional): A pandas DataFrame containing the dataset. 
                          If None, the dataset is loaded from 'datasets/heart.csv' 
                          and duplicates are dropped. Default is None.

        Attributes:
        ----------
        dataset (pd.DataFrame): The dataset used for training and evaluation.
        predicted_attr (str): The attribute to be predicted. Default is "target".
        max_iter (int): Maximum number of iterations for the model. Default is 2000.
        n_estimators (int): Number of estimators for ensemble methods. Default is 20.
        random_state (int): Random seed for reproducibility. Default is 0.
        max_depth (int): Maximum depth of the tree. Default is 7.
        criterion (str): Criterion for decision tree splitting. Default is "entropy".
        positive_outcome (int): The value representing a positive outcome. Default is 0.
        protected_attr (list): List of protected attributes. Default is ["sex"].
        num_repetitions (int): Number of repetitions for experiments. Default is 10.
        protected_attr_mappings (dict): Mappings for protected attributes. Default is {"sex": {"Female": 0, "Male": 1}}.

        """
        super().__init__()
        self.dataset = (
            dataset
            if dataset is not None
            else pd.read_csv("datasets/heart.csv").drop_duplicates()
        )
        self.predicted_attr = "target"
        self.max_iter = 2000
        self.n_estimators = 20
        self.random_state = 0
        self.max_depth = 7
        self.criterion = "entropy"
        self.positive_outcome = 0
        self.protected_attr = ["sex"]
        self.num_repetitions = 10
        self.protected_attr_mappings = {"sex": {"Female": [0], "Male": [1]}}

    def get_metrics(self, df_train):
        d = self.evaluate_metrics("sex", 1, "cp", df_train)
        d.update(self.evaluate_metrics("sex", 1, "thal", df_train, True))
        return d
