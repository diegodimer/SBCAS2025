import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Datasets.GliomaDataset import GliomaDataset
from experiment_utils import (evaluate_train_and_test_sets, feature_importante,
                              generate_charts, get_full_sets_graphs)


def remove_instances_2(x, conditions: list, percentage: float):
    new_x = x.loc[np.logical_and.reduce(conditions)]
    new_x_size = len(new_x)
    drop_indices = np.random.choice(
        new_x.index, min(percentage, new_x_size) if percentage >= 1 else min(round(new_x_size*percentage), new_x_size), replace=False)
    new_xtrain = x.drop(drop_indices)
    return new_xtrain

def gen_graph_for_sets(h: GliomaDataset, name: str):
    full_dataset_test = get_full_sets_graphs(h, name)
    generate_charts(h, name, full_dataset_test)
    evaluate_train_and_test_sets(h, name)

    plt.close("all")

    feature_importante(name, h)

def original_dataset():
    h = GliomaDataset()
    print("==========Original Dataset===========")

    acc, f1 = h.execute_models()
    global all_acs
    all_acs += acc
    global all_f1s
    all_f1s += f1
    gen_graph_for_sets(h, "original-dataset")
    return h.num_models()

def high_imbalance():
    def perturbe(X_train, y_train):
        print(f"Original size: {len(X_train)}")
        new_x_train = X_train.reset_index()
        new_x_train[h.predicted_attr] = y_train.reset_index()[h.predicted_attr]
        new_x_train = remove_instances_2(new_x_train, [new_x_train['Gender'] == 0, new_x_train['Grade'] == 1], .70)
        new_x_train = remove_instances_2(new_x_train, [new_x_train['Gender'] == 1, new_x_train['Grade'] == 0], 0.75)

        new_x_train = remove_instances_2(new_x_train, [new_x_train['Race'] == 0, new_x_train['Grade'] == 0], .75)
        new_x_train = remove_instances_2(new_x_train, [new_x_train['Race'] == 1, new_x_train['Grade'] == 1], .55)

        new_y_train = new_x_train[h.predicted_attr]
        new_x_train = new_x_train.drop(h.predicted_attr, axis=1)
        new_x_train = new_x_train.drop('index', axis=1)

        print(f"New size: {len(new_x_train)}")
        return new_x_train, new_y_train

    print("==========High Imbalance==========")
    print(
        "Remove 70% of women with Grade 1 and 75% of men with Grade 0, respectively\n"
        "Remove 75% of non-white people with Grade 0 and 55% of white people with Grade 1, respectively\n"
    )
    h = GliomaDataset()
    h.dropper = True
    h.perturbe = perturbe
    acc, f1 = h.execute_models()
    global all_acs
    all_acs += acc
    global all_f1s
    all_f1s += f1

    gen_graph_for_sets(h, "high-imbalance")

def equal_balance():
    def perturbe(X_train, y_train):
        print(f"Original size: {len(X_train)}")
        new_x_train = X_train.reset_index(drop=True)
        new_x_train[h.predicted_attr] = y_train.reset_index()[h.predicted_attr]

        # Create bins for race and grade
        race_bins = pd.cut(new_x_train['Race'], bins=2, labels=False)
        grade_bins = pd.cut(new_x_train['Grade'], bins=2, labels=False)

        # Get the minimum count for balancing within each bin combination
        min_count = new_x_train.groupby([race_bins, grade_bins]).size().min()
        balanced_x_train = new_x_train.groupby([race_bins, grade_bins]).apply(lambda x: x.sample(min_count)).reset_index(drop=True)
        
        # Remove some from the privileged group
        balanced_x_train = remove_instances_2(balanced_x_train, [balanced_x_train['Gender'] == 1, balanced_x_train['Race'] == 1], 3)
        print(f"Balanced size: {len(balanced_x_train)}")
        return balanced_x_train.drop(h.predicted_attr, axis=1), balanced_x_train[h.predicted_attr]

    print("==========Equal Balance==========")

    h = GliomaDataset()
    h.dropper = True
    h.perturbe = perturbe
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
