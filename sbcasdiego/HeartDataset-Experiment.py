import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Datasets.HeartDataset import HeartDataset
from experiment_utils import (evaluate_train_and_test_sets, feature_importante,
                              generate_charts, get_full_sets_graphs)
from tabulate import tabulate


def gen_graph_for_sets(h: HeartDataset, name: str):
    full_dataset_test = get_full_sets_graphs(h, name)
    generate_charts(h, name, full_dataset_test)
    evaluate_train_and_test_sets(h, name)

    plt.close("all")

    feature_importante(name, h)

def remove_instances(x, target, value, sex=0):
    new_x = x.loc[(x["sex"] == sex) & (x["target"] == target)]
    drop_indices = np.random.choice(
        new_x.index, value if value >= 1 else round(len(new_x) * value), replace=False
    )
    new_xtrain = x.drop(drop_indices)
    return new_xtrain


def remove_instances_2(x, conditions, value):
    new_x = x.loc[np.logical_and.reduce(conditions)]
    new_x_size = len(new_x)
    drop_indices = np.random.choice(
        new_x.index,
        min(value, new_x_size)
        if value >= 1
        else min(round(new_x_size * value), new_x_size),
        replace=False,
    )
    new_xtrain = x.drop(drop_indices)
    return new_xtrain


def original_dataset():
    global train_size_hi
    train_size_hi = 0
    def perturbe(X_train, y_train):
        global train_size_hi
        train_size_hi += len(X_train)
        return X_train, y_train
    h = HeartDataset()
    h.perturbe = perturbe
    h.dropper = True
    print("==========Original Dataset===========")

    acc, f1 = h.execute_models()
    global all_acs
    global all_f1s
    all_acs['Original Dataset'] = acc
    all_f1s['Original Dataset'] = f1
    print(f"Mean Train size: {train_size_hi/10}")
    gen_graph_for_sets(h, "original-dataset")
    return h.num_models()


def high_imbalance():
    global train_size_hi
    train_size_hi = 0
    def perturbe(x_train, y_train):
        new_x_train = x_train.reset_index()
        new_x_train[h.predicted_attr] = y_train.reset_index()[h.predicted_attr]

        new_x_train = remove_instances_2(
            new_x_train,
            [
                new_x_train["thal"] == 2,
                new_x_train["target"] == 0,
                new_x_train["sex"] == 0,
            ],
            0.85,
        )
        new_x_train = remove_instances_2(
            new_x_train,
            [
                new_x_train["thal"] == 3,
                new_x_train["target"] == 0,
                new_x_train["sex"] == 0,
            ],
            0.80,
        )

        new_x_train = remove_instances_2(
            new_x_train,
            [
                new_x_train["cp"] == 2,
                new_x_train["target"] == 0,
                new_x_train["sex"] == 0,
            ],
            0.80,
        )
        new_x_train = remove_instances_2(
            new_x_train,
            [
                new_x_train["cp"] == 0,
                new_x_train["target"] == 0,
                new_x_train["sex"] == 0,
            ],
            0.80,
        )
        new_x_train = remove_instances(new_x_train, 1, 0.2)
        new_y_train = new_x_train[h.predicted_attr]
        new_x_train = new_x_train.drop(h.predicted_attr, axis=1)
        new_x_train = new_x_train.drop("index", axis=1)
        global train_size_hi
        train_size_hi += len(new_x_train)
        return new_x_train, new_y_train

    print("==========High Imbalance==========")
    print(
        "Removed 85% of women with thal=2 and negative output, 80% of women with thal=3 and negative output, "
        "80% of women with cp=2 and negative output, 80% of women with cp=0 and negative output, "
        "and 20% of instances with positive output"
    )
    h = HeartDataset()
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
        complete_x_train = x_train.reset_index()
        complete_x_train[h.predicted_attr] = y_train.reset_index()[h.predicted_attr]

        positive_out_in_train = len(
            complete_x_train.loc[
                (complete_x_train["sex"] == 1)
                & (complete_x_train[h.predicted_attr] == 1)
            ]
        ) - len(
            complete_x_train.loc[
                (complete_x_train["sex"] == 0)
                & (complete_x_train[h.predicted_attr] == 1)
            ]
        )
        negative_out_in_train = len(
            complete_x_train.loc[
                (complete_x_train["sex"] == 1)
                & (complete_x_train[h.predicted_attr] == 0)
            ]
        ) - len(
            complete_x_train.loc[
                (complete_x_train["sex"] == 0)
                & (complete_x_train[h.predicted_attr] == 0)
            ]
        )

        complete_x_train = remove_instances(
            complete_x_train, 1, positive_out_in_train, sex=1
        )
        new_x_train = remove_instances(
            complete_x_train, 0, negative_out_in_train, sex=1
        )
        new_y_train = new_x_train[h.predicted_attr]
        new_x_train = new_x_train.drop(h.predicted_attr, axis=1)
        new_x_train = new_x_train.drop("index", axis=1)
        global train_size_eq
        train_size_eq += len(new_x_train)
        return new_x_train, new_y_train

    print("==========Equally Balanced==========")
    h = HeartDataset()
    h.dropper = True
    h.gen_graph()
    h.perturbe = perturbe
    acc, f1 = h.execute_models()
    global all_acs
    global all_f1s
    all_acs["Equal Balance"] = acc
    all_f1s["Equal Balance"] = f1
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
latex_table = df.style.to_latex(caption='Performance results for the Heart Dataset',  position='p')

h = HeartDataset()
with open(f"results/{type(h).__name__}/performance_results.tex", "w") as f:
    f.write(latex_table)
