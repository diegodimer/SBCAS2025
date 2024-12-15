import numpy as np
import pandas as pd
from clarify_helper import pdfs_aligned_nonzero
from numpy import Infinity


class PreTrainingBias:
    def _class_imbalance(self, n_a, n_d):
        return (n_a - n_d) / (n_a + n_d)

    def _difference_in_positive_proportions_of_labels(self, q_a, q_d):
        return q_a - q_d

    def _kl_divergence(self, p, q):
        return np.sum(p * np.log(p / q))

    def _divide(self, a, b) -> float:
        if b == 0 and a == 0:
            return 0.0
        if b == 0:
            if a < 0:
                return -Infinity
            return Infinity
        return a / b

    def class_imbalance(self, df, label, threshold=None):
        facet_counts = df[label].value_counts(sort=True)
        if len(facet_counts) == 2:
            return self._class_imbalance(facet_counts.values[0], facet_counts.values[1])
        else:  # is not a binary attr
            if threshold is None:
                raise Exception("Threshold not defined")
            a = len(df[df[label] > threshold])
            b = len(df[df[label] <= threshold])
            return self._class_imbalance(max(a, b), min(a, b))

    def class_imbalance_per_label(self, df, label, privileged_group) -> float:
        return self._class_imbalance(
            (df[label].values == privileged_group).sum(),
            (df[label].values != privileged_group).sum(),
        )

    def kl_divergence(
        self, df, target, protected_attribute: str, privileged_group
    ) -> float:
        label = df[target]
        p_list = list()
        sensitive_facet_index = df[protected_attribute] != privileged_group
        unsensitive_facet_index = df[protected_attribute] == privileged_group
        p_list = pdfs_aligned_nonzero(
            label[unsensitive_facet_index], label[sensitive_facet_index]
        )
        ks_val = 0
        for i, j in enumerate(p_list[0]):  # j = 0, 2 , i = 0
            ks_val += self._kl_divergence(j, p_list[1][i])
        return ks_val

    def ks(self, df, target, protected_attribute: str, privileged_group) -> float:
        label = df[target]
        p_list = list()
        sensitive_facet_index = df[protected_attribute] != privileged_group
        unsensitive_facet_index = df[protected_attribute] == privileged_group
        p_list = pdfs_aligned_nonzero(
            label[unsensitive_facet_index], label[sensitive_facet_index]
        )
        ks_val = 0
        for i, j in enumerate(p_list[0]):
            ks_val = max(ks_val, abs(np.subtract(j, p_list[1][i])))
        return ks_val

    def cddl(
        self,
        df: pd.DataFrame,
        target: str,
        positive_outcome,
        protected_attribute,
        privileged_group,
        group_variable,
    ) -> float:
        unique_groups = np.unique(df[group_variable])
        cdd = np.array([])
        counts = np.array([])
        for subgroup_variable in unique_groups:
            counts = np.append(
                counts, (df[group_variable].values == subgroup_variable).sum()
            )
            num_a = len(
                df[
                    (df[target] == positive_outcome)
                    & (df[protected_attribute] != privileged_group)
                    & (df[group_variable] == subgroup_variable)
                ]
            )
            denom_a = len(
                df[
                    (df[target] == positive_outcome)
                    & (df[group_variable] == subgroup_variable)
                ]
            )
            a = num_a / denom_a if denom_a != 0 else 0
            num_d = len(
                df[
                    (df[target] != positive_outcome)
                    & (df[protected_attribute] != privileged_group)
                    & (df[group_variable] == subgroup_variable)
                ]
            )
            denom_d = len(
                df[
                    (df[target] != positive_outcome)
                    & (df[group_variable] == subgroup_variable)
                ]
            )
            d = num_d / denom_d if denom_d != 0 else 0
            cdd = np.append(cdd, d - a)
        return self._divide(np.sum(counts * cdd), np.sum(counts))

    def global_evaluation(
        self,
        df: pd.DataFrame,
        target: str,
        positive_outcome,
        protected_attribute,
        privileged_group,
        group_variable,
    ):
        dic = {
            f"Class Imbalance ({protected_attribute})": self.class_imbalance_per_label(
                df, protected_attribute, privileged_group
            ),
            f"KL Divergence ({protected_attribute})": self.kl_divergence(
                df, target, protected_attribute, privileged_group
            ),
            f"KS ({protected_attribute})": self.ks(
                df, target, protected_attribute, privileged_group
            ),
            f"CDDL ({protected_attribute}, {group_variable})": self.cddl(
                df,
                target,
                positive_outcome,
                protected_attribute,
                privileged_group,
                group_variable,
            ),
        }
        return dic
