import numpy as np
import pandas as pd
from functional import seq



class PostTrainingBias:
   
    def _DPPL(self, df: pd.DataFrame, y_hat:str, protected_attr:str, privileged_group: str) -> float:
        q_p = len(df[(df[y_hat] == 1) & (df[protected_attr] == privileged_group)]) / len(df[df[protected_attr] == privileged_group])
        q_d = len(df[(df[y_hat] == 1) & (df[protected_attr] != privileged_group)]) / len(df[df[protected_attr] != privileged_group])

        return q_p - q_d
   
    def _DI(self,df: pd.DataFrame, y_hat:str, protected_attr:str, privileged_group: str) -> float:
        q_p = len(df[(df[y_hat] == 1) & (df[protected_attr] == privileged_group)]) / len(df[df[protected_attr] == privileged_group])
        q_d = len(df[(df[y_hat] == 1) & (df[protected_attr] != privileged_group)]) / len(df[df[protected_attr] != privileged_group])

        return q_d / q_p
   

    def _DCA_DCR(self, df: pd.DataFrame, y: str, y_hat: str, protected_attr: str, p: int) -> float:
       n_p_1 = len(df[(df[y] == 1) & (df[protected_attr] == p)])
       n_p_0 = len(df[(df[y] == 0) & (df[protected_attr] == p)])
       n_d_1 = len(df[(df[y] == 1) & (df[protected_attr] != p)])
       n_d_0 = len(df[(df[y] == 0) & (df[protected_attr] != p)])
       n_hat_p_1 = len(df[(df[y_hat] == 1) & (df[protected_attr] == p)])
       n_hat_p_0 = len(df[(df[y_hat] == 0) & (df[protected_attr] == p)])
       n_hat_d_1 = len(df[(df[y_hat] == 1) & (df[protected_attr] != p)])
       n_hat_d_0 = len(df[(df[y_hat] == 0) & (df[protected_attr] != p)])
       DCA = (
           (n_p_1 / n_hat_p_1) -
           (n_d_1 / n_hat_d_1)
       )
       DCR = (
           (n_p_0 / n_hat_p_0) -
           (n_d_0 / n_hat_d_0)
       )
       return DCA, DCR

    def global_evaluation(
        self,
        df: pd.DataFrame,
        y: str,
        y_hat: str,
        protected_attribute: str,
        privileged_group: str,
    ):
        DCA_DCR = self._DCA_DCR(df, y, y_hat, protected_attribute, privileged_group)
        dic = {
            f"DPPL ({protected_attribute})": self._DPPL(df, y_hat, protected_attribute, privileged_group),
            f"DI ({protected_attribute})": self._DI(df, y_hat, protected_attribute, privileged_group),
            f"DCA ({protected_attribute})": DCA_DCR[0],
            f"DCR ({protected_attribute})": DCA_DCR[1]
        }
        return dic
