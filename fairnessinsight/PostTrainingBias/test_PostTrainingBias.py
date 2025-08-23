import pandas as pd

from fairnessinsight.PostTrainingBias.PostTrainingBias import PostTrainingBias

# Heart Disease dataset extracted from
# https://www.kaggle.com/datasets/rishidamarla/heart-disease-prediction?resource=download

df = pd.read_csv(
    "fairnessinsight/PreTrainingBias/Heart_Disease_Prediction.csv")
df["y"] = df["Heart Disease"].apply(
    lambda x: 1 if x == "Absence" else 0)
df["y_hat"] = df["Heart Disease"].apply(lambda x: 1 if x == "Absence" else 0)
pt = PostTrainingBias()
df_only_male = df.loc[df["Sex"] == 1]
df_only_female = df.loc[df["Sex"] == 0]


def test_dppl():
    # Test for Demographic Parity Difference (DPPL). From the book chapter
    df_mixed = pd.concat(
        [
            df_only_female.loc[df_only_female["y_hat"] == 0].sample(5),
            df_only_female.loc[df_only_female["y_hat"] == 1].sample(5),
            df_only_male.loc[df_only_male["y_hat"] == 0].sample(3),
            df_only_male.loc[df_only_male["y_hat"] == 1].sample(7),
        ]
    )
    result = pt._DPPL(df_mixed, "y_hat", "Sex", 1)
    assert round(result, 4) == 0.2
    del df_mixed


def test_di():
    # Test for Disparate Impact (DI). From the book chapter
    df_mixed = pd.concat(
        [
            df_only_female.loc[df_only_female["y_hat"] == 0].sample(5),
            df_only_female.loc[df_only_female["y_hat"] == 1].sample(5),
            df_only_male.loc[df_only_male["y_hat"] == 0].sample(3),
            df_only_male.loc[df_only_male["y_hat"] == 1].sample(7),
        ]
    )
    result = pt._DI(df_mixed, "y_hat", "Sex", 1)
    assert round(result, 4) == 0.7143
    del df_mixed


def test_dca_dcr():
    # Test for Difference in Conditional Acceptance/Rejection Rates (DCA/DCR)
    # DCA suggests bias favoring women (70/60 vs 20/30 | 17% less for men vs 33% more for women)
    # DCR suggests bias against women (30/40 vs 30/20 | 25% less vs 50% more for women)
    df_mixed = pd.DataFrame({
        "y": [1] * 70 + [0] * 30 + [1] * 20 + [0] * 30,
        "y_hat": [1] * 60 + [0] * 40 + [1] * 30 + [0] * 20,
        "Sex": [1] * 100 + [0] * 50,
    })

    dca, dcr = pt._DCA_DCR(df_mixed, "y", "y_hat", "Sex", 1)
    assert round(dca, 4) == 0.5000
    assert round(dcr, 4) == -0.75
    del df_mixed


def test_global_evaluation():
    # Test for global evaluation metrics
    df_mixed = pd.concat(
        [
            df_only_female.loc[df_only_female["y_hat"] == 0].sample(6),
            df_only_female.loc[df_only_female["y_hat"] == 1].sample(4),
            df_only_male.loc[df_only_male["y_hat"] == 0].sample(4),
            df_only_male.loc[df_only_male["y_hat"] == 1].sample(6),
        ]
    )
    result = pt.global_evaluation(
        df_mixed, "y", "y_hat", "Sex", 1)
    assert "DPPL (Sex)" in result
    assert "DI (Sex)" in result
    assert "DCA (Sex)" in result
    assert "DCR (Sex)" in result
    del df_mixed
