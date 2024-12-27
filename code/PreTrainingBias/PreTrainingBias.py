import numpy as np
import pandas as pd
from functional import seq
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
        if not isinstance(privileged_group, list):
            privileged_group = [privileged_group]
        return self._class_imbalance(
            df[label].isin(privileged_group).sum(),
            (~df[label].isin(privileged_group)).sum(),
        )

    def kl_divergence(
        self, df, target, protected_attribute: str, privileged_group
    ) -> float:
        if not isinstance(privileged_group, list):
            privileged_group = [privileged_group]
        label = df[target]
        p_list = list()
        sensitive_facet_index = ~df[protected_attribute].isin(privileged_group)
        unsensitive_facet_index = df[protected_attribute].isin(privileged_group)
        p_list = self.pdfs_aligned_nonzero(
            label[unsensitive_facet_index], label[sensitive_facet_index]
        )
        ks_val = 0
        for i, j in enumerate(p_list[0]):  # j = 0, 2 , i = 0
            ks_val += self._kl_divergence(j, p_list[1][i])
        return ks_val

    def ks(self, df, target, protected_attribute: str, privileged_group) -> float:
        if not isinstance(privileged_group, list):
            privileged_group = [privileged_group]
        label = df[target]
        p_list = list()
        sensitive_facet_index = ~df[protected_attribute].isin(privileged_group)
        unsensitive_facet_index = df[protected_attribute].isin(privileged_group)
        p_list = self.pdfs_aligned_nonzero(
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
        if not isinstance(privileged_group, list):
            privileged_group = [privileged_group]
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
                    & (~df[protected_attribute].isin(privileged_group))
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
                    & (~df[protected_attribute].isin(privileged_group))
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

    # Code borrowed from https://github.com/aws/amazon-sagemaker-clarify/blob/53cb4172bea1efd673b6d48c3a006ce4ac1fd5a5/src/smclarify/util/__init__.py
    # Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.


    def pdf(self, xs) -> dict:
        """
        Probability distribution function.
        
        :param xs: input sequence
        :return: sequence of tuples as (value, frequency)
        """
        counts = seq(xs).map(lambda x: (x, 1)).reduce_by_key(lambda x, y: x + y)
        total = counts.map(lambda x: x[1]).sum()
        result_pdf = counts.map(lambda x: (x[0], x[1] / total)).sorted().list()
        return result_pdf


    def pdfs_aligned_nonzero(self, *args) -> list[np.ndarray]:
        """
        Convert a list of discrete pdfs / freq counts to aligned numpy arrays of the same size for common non-zero elements.

        :return: pair of numpy arrays of the same size with the aligned pdfs.
        """
        num_pdfs = len(args)
        pdfs = []
        for x in args:
            pdfs.append(self.pdf(x))

        def keys(_xs):
            return seq(_xs).map(lambda x: x[0])

        # Extract union of keys
        all_keys = seq(pdfs).flat_map(keys).distinct().sorted()

        # Index all pdfs by value
        dict_pdfs = seq(pdfs).map(dict).list()

        # ADDED BY ME: if key is not in the dict_pdfs, it should be with 0 probability
        for i in all_keys:
            for j in dict_pdfs:
                if i not in j:
                    j[i] = 0.0

        # result aligned lists
        aligned_lists: list[list] = [[] for x in range(num_pdfs)]

        # fill keys present in all pdfs
        # OTHER CHANGE BY ME: if key is not present, it should have 0 probability. KS results would not be precise if not
        for i, key in enumerate(all_keys):
            # for j, d in enumerate(dict_pdfs):
            #     if d.get(key, 0) == 0:
            #         break
            # else:
            # All keys exist and are != 0
            for j, d in enumerate(dict_pdfs):
                aligned_lists[j].append(d[key])
        np_arrays = seq(aligned_lists).map(np.array).list()
        return np_arrays
