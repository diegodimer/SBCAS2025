import matplotlib.pyplot as plt
import numpy as np
from experiment_utils import (evaluate_train_and_test_sets, feature_importante,
                              generate_pies, get_full_sets_graphs,
                              remove_instances)
from Datasets.IntersectionalBiasDataset import IntersectionalBiasDataset


def gen_graph_for_sets(h: IntersectionalBiasDataset, name: str):
    full_dataset_test = get_full_sets_graphs(h, name)
    generate_pies(h, name, full_dataset_test)
    evaluate_train_and_test_sets(h, name)

    # fig1, ax1 = plt.subplots()
    # ax1.set_title('Box Plot for Metric Values')
    # pd.DataFrame(metrics_all).boxplot(ax=ax1, rot=45)
    # fig1.savefig("boxplot-metrics.png")
    plt.close("all")

    feature_importante(name, h)


def original_dataset():
    h = IntersectionalBiasDataset()
    print("==========Original Dataset===========")

    acc, f1 = h.execute_models()
    global all_acs
    all_acs += acc
    global all_f1s
    all_f1s += f1
    gen_graph_for_sets(h, "original-dataset")
    return h.num_models()

def high_imbalance():
    def perturbe(x_train, y_train):
        new_x_train = x_train.reset_index()
        new_x_train[h.predicted_attr] = y_train.reset_index()[h.predicted_attr]

        new_x_train = remove_instances(
            new_x_train, [new_x_train["Race"] == 0, new_x_train["Diagnosis"] == 0], 0.95
        )
        new_x_train = remove_instances(
            new_x_train, [new_x_train["Race"] == 0, new_x_train["Diagnosis"] == 1], 0.5
        )
        new_x_train = remove_instances(
            new_x_train, [new_x_train["Sex"] == 0, new_x_train["Diagnosis"] == 0], 0.8
        )
        new_x_train = remove_instances(
            new_x_train, [new_x_train["Sex"] == 0, new_x_train["Diagnosis"] == 1], 0.95
        )

        new_y_train = new_x_train[h.predicted_attr]
        new_x_train = new_x_train.drop(h.predicted_attr, axis=1)
        new_x_train = new_x_train.drop("index", axis=1)

        return new_x_train, new_y_train

    print("==========High Imbalance==========")
    print(
        "Remove 95% of instances of non-white with negative output, 50% of non-white with positive output,  80% of women with negative output, 95% of women with positive output"
    )
    h = IntersectionalBiasDataset()
    h.dropper = True
    h.perturbe = perturbe
    acc, f1 = h.execute_models()
    global all_acs
    all_acs += acc
    global all_f1s
    all_f1s += f1

    gen_graph_for_sets(h, "high-imbalance")


def equal_balance():
    def perturbe(x_train, y_train):
        new_x_train = x_train.reset_index()
        new_x_train[h.predicted_attr] = y_train.reset_index()[h.predicted_attr]

        attribute_combinations = np.array(
            np.meshgrid([0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1])
        ).T.reshape(-1, 6)

        for attr_comb in attribute_combinations:
            var11 = len(
                new_x_train.loc[
                    (new_x_train["Sex"] == attr_comb[0])
                    & (new_x_train["Diagnosis"] == attr_comb[1])
                    & (new_x_train["Race"] == attr_comb[2])
                ]
            )
            var12 = len(
                new_x_train.loc[
                    (new_x_train["Sex"] == attr_comb[3])
                    & (new_x_train["Diagnosis"] == attr_comb[4])
                    & (new_x_train["Race"] == attr_comb[5])
                ]
            )
            if var11 > var12:
                new_x_train = remove_instances(
                    new_x_train,
                    [
                        new_x_train["Sex"] == attr_comb[0],
                        new_x_train["Diagnosis"] == attr_comb[1],
                        new_x_train["Race"] == attr_comb[2],
                    ],
                    var11 - var12,
                )
            elif var12 > var11:
                new_x_train = remove_instances(
                    new_x_train,
                    [
                        new_x_train["Sex"] == attr_comb[3],
                        new_x_train["Diagnosis"] == attr_comb[4],
                        new_x_train["Race"] == attr_comb[5],
                    ],
                    var12 - var11,
                )

        new_y_train = new_x_train[h.predicted_attr]
        new_x_train = new_x_train.drop(h.predicted_attr, axis=1)
        new_x_train = new_x_train.drop("index", axis=1)

        return new_x_train, new_y_train

    print("==========Equally Balanced==========")
    h = IntersectionalBiasDataset()
    h.dropper = True
    h.perturbe = perturbe
    h.gen_graph()
    acc, f1 = h.execute_models()
    global all_acs
    all_acs += acc
    global all_f1s
    all_f1s += f1
    gen_graph_for_sets(h, "equal-balance")


all_acs = []
all_f1s = []
n_models = original_dataset()
high_imbalance()
equal_balance()
print("\nLaTeX Table for Accuracy")
for i in range(n_models):
    print(
        f" & {all_acs[i]: >2.3f} & {all_acs[i + n_models]: >2.3f} & {all_acs[i + n_models + n_models]: >2.3f} ",
        end="",
    )
    print("\n")
print("\nLaTeX Table for F1")

for i in range(n_models):
    print(
        f" & {all_f1s[i]: >2.3f} & {all_f1s[i + n_models]: >2.3f} & {all_f1s[i + n_models + n_models]: >2.3f} ",
        end="",
    )
    print("\n")

print("avg acc (all models)")
for i in range(3):
    idx = i * n_models
    print(round(sum(all_acs[idx : idx + n_models]) / n_models, 3))

print("avg f1 (all models)")
for i in range(3):
    idx = i * n_models
    print(round(sum(all_f1s[idx : idx + n_models]) / n_models, 3))
