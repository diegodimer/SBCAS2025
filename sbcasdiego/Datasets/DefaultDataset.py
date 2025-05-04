import pandas as pd
import numpy as np
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


    def get_metrics_graph(self, metrics: dict, plt, rotate_x_labels=False):
        kl_divergence_keys = list(metrics['KL Divergence'].keys())

        for value in kl_divergence_keys:
            if np.isnan(metrics['KL Divergence'][value]):
                metrics['KL Divergence'][value] = 0
                metrics['KL Divergence'][f"{value}*"] = 0
                del metrics['KL Divergence'][value]

        if rotate_x_labels:
            fig, axes = plt.subplots(2, 2, figsize=(30, 12))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        for idx, (metric_name, metric_values) in enumerate(metrics.items()):         
            ax = axes[idx]
            sorted_metric_values = dict(sorted(metric_values.items(), key=lambda item: item[1]))
            ax.bar(sorted_metric_values.keys(), sorted_metric_values.values())
            ax.set_title(metric_name)
            ax.set_ylabel(metric_name)
            ax.set_xlabel("Class")
            ax.axhline(y=0, color='k')
            if rotate_x_labels:
                ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for better visibility

        plt.tight_layout()  # Adjust layout to prevent overlap

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
