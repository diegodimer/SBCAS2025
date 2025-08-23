import numpy as np
import pandas as pd


class PostTrainingBias:
    def safe_division(self, numerator, denominator):
        """
        Safely performs division, handling cases where the denominator is zero.

        Parameters:
        numerator (float): The numerator of the division.
        denominator (float): The denominator of the division.

        Returns:
        float: The result of the division, or 0.0/np.inf if the denominator is zero.
        """
        if denominator == 0:
            return 0.0 if numerator == 0 else np.inf
        return numerator / denominator

    def _DPPL(
            self,
            df: pd.DataFrame,
            y_hat: str,
            protected_attr: str,
            privileged_group: str) -> float:
        """
        Calculates the Difference in Positive Proportions in Predicted Labels (DPPL) between the privileged group
        and the unprivileged group.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        y_hat (str): The column name for the model's predictions.
        protected_attr (str): The column name for the protected attribute.
        privileged_group (str): The value of the protected attribute representing the privileged group.

        Returns:
        float: The DPPL value.
        """
        q_p = len(df[(df[y_hat] == 1) & (df[protected_attr] == privileged_group)]
                  ) / len(df[df[protected_attr] == privileged_group])
        q_d = len(df[(df[y_hat] == 1) & (df[protected_attr] != privileged_group)]
                  ) / len(df[df[protected_attr] != privileged_group])

        return q_p - q_d

    def _DI(
            self,
            df: pd.DataFrame,
            y_hat: str,
            protected_attr: str,
            privileged_group: str) -> float:
        """
        Calculates the Disparate Impact (DI), which is the ratio of positive prediction rates between the unprivileged group
        and the privileged group.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        y_hat (str): The column name for the model's predictions.
        protected_attr (str): The column name for the protected attribute.
        privileged_group (str): The value of the protected attribute representing the privileged group.

        Returns:
        float: The DI value.
        """
        q_p = len(df[(df[y_hat] == 1) & (df[protected_attr] == privileged_group)]
                  ) / len(df[df[protected_attr] == privileged_group])
        q_d = len(df[(df[y_hat] == 1) & (df[protected_attr] != privileged_group)]
                  ) / len(df[df[protected_attr] != privileged_group])

        return self.safe_division(q_d, q_p)

    def _DCA_DCR(
            self,
            df: pd.DataFrame,
            y: str,
            y_hat: str,
            protected_attr: str,
            p: int) -> float:
        """
        Calculates the Difference in Conditional Acceptance Rates (DCA) and Difference in Conditional Rejection Rates (DCR).

        Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        y (str): The column name for the observed labels.
        y_hat (str): The column name for the model's predictions.
        protected_attr (str): The column name for the protected attribute.
        p (int): The value of the protected attribute representing the privileged group.

        Returns:
        tuple: A tuple containing the DCA and DCR values.
        """
        n_p_1 = len(df[(df[y] == 1) & (df[protected_attr] == p)])
        n_p_0 = len(df[(df[y] == 0) & (df[protected_attr] == p)])
        n_d_1 = len(df[(df[y] == 1) & (df[protected_attr] != p)])
        n_d_0 = len(df[(df[y] == 0) & (df[protected_attr] != p)])
        n_hat_p_1 = len(df[(df[y_hat] == 1) & (df[protected_attr] == p)])
        n_hat_p_0 = len(df[(df[y_hat] == 0) & (df[protected_attr] == p)])
        n_hat_d_1 = len(df[(df[y_hat] == 1) & (df[protected_attr] != p)])
        n_hat_d_0 = len(df[(df[y_hat] == 0) & (df[protected_attr] != p)])

        DCA = self.safe_division(n_p_1, n_hat_p_1) - \
            self.safe_division(n_d_1, n_hat_d_1)
        DCR = self.safe_division(n_p_0, n_hat_p_0) - \
            self.safe_division(n_d_0, n_hat_d_0)
        return DCA, DCR

    def global_evaluation(
            self,
            df: pd.DataFrame,
            y: str,
            y_hat: str,
            protected_attribute: str,
            privileged_group: str):
        """
        Provides a global evaluation of fairness metrics, including DPPL, DI, DCA, and DCR.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        y (str): The column name for the observed labels.
        y_hat (str): The column name for the model's predictions.
        protected_attribute (str): The column name for the protected attribute.
        privileged_group (str): The value of the protected attribute representing the privileged group.

        Returns:
        dict: A dictionary containing the calculated fairness metrics.
        """
        DCA_DCR = self._DCA_DCR(
            df,
            y,
            y_hat,
            protected_attribute,
            privileged_group)
        dic = {
            f"DPPL ({protected_attribute})": self._DPPL(
                df,
                y_hat,
                protected_attribute,
                privileged_group),
            f"DI ({protected_attribute})": self._DI(
                df,
                y_hat,
                protected_attribute,
                privileged_group),
            f"DCA ({protected_attribute})": DCA_DCR[0],
            f"DCR ({protected_attribute})": DCA_DCR[1],
        }
        return dic
