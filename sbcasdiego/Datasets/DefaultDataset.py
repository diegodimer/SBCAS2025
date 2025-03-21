import pandas as pd

from sbcasdiego.Datasets.BaseDataset import BaseDataset


class DefaultDataset(BaseDataset):

    def __init__(
        self,
        dataset: pd.DataFrame,
        predicted_attr: str,
        protected_attr: list[str],
        protected_attr_mappings: dict,
        positive_outcome: int,
        criterion="entropy",
        max_iter=2000,
        n_estimators=20,
        random_state=0,
        max_depth=7,
        num_repetitions=10,
    ):
        super().__init__()
        self.dataset = dataset
        self.predicted_attr = predicted_attr
        self.max_iter = max_iter
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_depth = max_depth
        self.criterion = criterion
        self.positive_outcome = positive_outcome
        self.protected_attr = protected_attr
        self.num_repetitions = num_repetitions
        self.protected_attr_mappings = protected_attr_mappings


    def get_metrics_graph(self, metrics: dict, metric_name):
        # plot the values for metric_name in dictionary metrics
        pass

    def get_metrics(self, correlated_vars, df_train: None):
        if df_train is None:
            df_train = self.dataset
        d = {}
        metrics_names = ['Class Imbalance', 'KS', 'KL Divergence', 'CDDL']
        for i in self.protected_attr:
                metrics = self.evaluate_metrics_per_attr(
                    i, 1, correlated_vars, df_train)
                for metric in metrics_names:
                    if metric not in d:
                        d[metric] = {}
                    d[metric][i] = metrics[metric][i]
                
        return d
