import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Datasets.IntersectionalBiasDataset import IntersectionalBiasDataset
from experiment_utils import (evaluate_train_and_test_sets, feature_importante,
                              generate_charts, get_full_sets_graphs,
                              remove_instances)
from tabulate import tabulate


def gen_graph_for_sets(h: IntersectionalBiasDataset, name: str):
    full_dataset_test = get_full_sets_graphs(h, name)
    generate_charts(h, name, full_dataset_test)
    evaluate_train_and_test_sets(h, name)

    plt.close("all")

    feature_importante(name, h)


def original_dataset():
    h = IntersectionalBiasDataset()
    print("==========Original Dataset===========")

    acc, f1 = h.execute_models()
    global all_acs
    global all_f1s
    all_acs['Original Dataset'] = acc
    all_f1s['Original Dataset'] = f1
    gen_graph_for_sets(h, "original-dataset")
    return h.num_models()

def high_imbalance():
    global train_size_hi
    train_size_hi = 0
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
        global train_size_hi
        train_size_hi += len(new_x_train)
        return new_x_train, new_y_train

    print("==========High Imbalance==========")
    print(
        "Remove 95% of instances of non-white with negative output"
        "Remove 50% of non-white with positive output"
        "Remove 80% of women with negative output and 95% of women with positive output"
    )
    h = IntersectionalBiasDataset()
    h.dropper = True
    h.perturbe = perturbe
    acc, f1 = h.execute_models()
    global all_acs
    global all_f1s
    all_acs['High Imbalance'] = acc
    all_f1s['High Imbalance'] = f1
    print(f"Mean Train size: {train_size_hi/10}")
    gen_graph_for_sets(h, "high-imbalance")


def equal_balance():
    global train_size_eq
    train_size_eq = 0
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
        global train_size_eq
        train_size_eq += len(new_x_train)
        return new_x_train, new_y_train

    print("==========Equally Balanced==========")
    h = IntersectionalBiasDataset()
    h.dropper = True
    h.perturbe = perturbe
    h.gen_graph()
    acc, f1 = h.execute_models()
    global all_acs
    global all_f1s
    all_acs['Equal Balance'] = acc
    all_f1s['Equal Balance'] = f1
    print(f"Mean Train size: {train_size_eq/10}")
    gen_graph_for_sets(h, "equal-balance")

all_acs = {}
all_f1s = {}
n_models = original_dataset()
high_imbalance()
equal_balance()
print("\nLaTeX Table for Accuracy")

data = {
    'Metric': ['Accuracy', 'Accuracy', 'Accuracy', 
               'F1-Score', 'F1-Score', 'F1-Score'],
    'Training Algorithm': ['Logistic Regression', 'Decision Tree', 'Random Forest',
                            'Logistic Regression', 'Decision Tree', 'Random Forest'],
    'Original Dataset': list(all_acs['Original Dataset'].values()) + list(all_f1s['Original Dataset'].values()),
    'High Imbalance': list(all_acs['High Imbalance'].values()) + list(all_f1s['High Imbalance'].values()),
    'Equal Balance': list(all_acs['Equal Balance'].values()) + list(all_f1s['Equal Balance'].values()),
}

df = pd.DataFrame(data)

print(tabulate(df, headers='keys', tablefmt='grid'))
df.set_index(list(data.keys()), inplace=True)
latex_table = df.style.to_latex(caption='Performance results for the Intersectional Bias Dataset',  position='p')

h = IntersectionalBiasDataset()
with open(f"results/{type(h).__name__}/performance_results.tex", "w") as f:
    f.write(latex_table)
