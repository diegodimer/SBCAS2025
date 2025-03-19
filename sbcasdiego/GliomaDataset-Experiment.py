
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Datasets.GliomaDataset import GliomaDataset
from experiment_utils import (evaluate_train_and_test_sets, feature_importante,
                              generate_charts, get_full_sets_graphs)
from tabulate import tabulate


def remove_instances_2(x, conditions: list, percentage: float):
    new_x = x.loc[np.logical_and.reduce(conditions)]
    new_x_size = len(new_x)
    drop_indices = np.random.choice(
        new_x.index, min(percentage, new_x_size)
        if percentage >= 1
        else min(round(new_x_size*percentage), new_x_size), replace=False)
    new_xtrain = x.drop(drop_indices)
    return new_xtrain


def gen_graph_for_sets(h: GliomaDataset, name: str):
    full_dataset_test = get_full_sets_graphs(h, name)
    generate_charts(h, name, full_dataset_test)
    evaluate_train_and_test_sets(h, name)

    plt.close("all")

    feature_importante(name, h)


def original_dataset():
    global train_size_hi
    train_size_hi = 0

    def perturbe(X_train, y_train):
        global train_size_hi
        train_size_hi += len(X_train)
        return X_train, y_train

    h = GliomaDataset()
    h.perturbe = perturbe
    h.dropper = True
    print("==========Original Dataset===========")

    acc, f1 = h.execute_models()
    global all_acs
    all_acs['Original Dataset'] = acc
    global all_f1s
    all_f1s['Original Dataset'] = f1
    print(f"Mean Train size: {train_size_hi/10}")
    gen_graph_for_sets(h, "original-dataset")
    return h.num_models()


def high_imbalance():
    global train_size
    train_size = 0

    def perturbe(X_train, y_train):
        new_x_train = X_train.reset_index()
        new_x_train[h.predicted_attr] = y_train.reset_index()[h.predicted_attr]
        new_x_train = remove_instances_2(
            new_x_train, [new_x_train['Gender'] == 0, new_x_train['Grade'] == 1], .70)
        new_x_train = remove_instances_2(
            new_x_train, [new_x_train['Gender'] == 1, new_x_train['Grade'] == 0], 0.75)

        new_x_train = remove_instances_2(
            new_x_train, [new_x_train['Race'] == 0, new_x_train['Grade'] == 0], .75)
        new_x_train = remove_instances_2(
            new_x_train, [new_x_train['Race'] == 1, new_x_train['Grade'] == 1], .55)

        new_y_train = new_x_train[h.predicted_attr]
        new_x_train = new_x_train.drop(h.predicted_attr, axis=1)
        new_x_train = new_x_train.drop('index', axis=1)
        global train_size
        train_size += len(new_x_train)
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
    global all_f1s
    all_acs['High Imbalance'] = acc
    all_f1s['High Imbalance'] = f1
    print(f"Mean Train size: {train_size/10}")
    gen_graph_for_sets(h, "high-imbalance")


def equal_balance():
    global train_size_eq
    train_size_eq = 0

    def perturbe(X_train, y_train):
        new_x_train = X_train.reset_index(drop=True)
        new_x_train[h.predicted_attr] = y_train.reset_index()[h.predicted_attr]

        # Create bins for race and grade
        race_bins = pd.cut(new_x_train['Race'], bins=2, labels=False)
        grade_bins = pd.cut(new_x_train['Grade'], bins=2, labels=False)

        # Get the minimum count for balancing within each bin combination
        min_count = new_x_train.groupby([race_bins, grade_bins]).size().min()
        balanced_x_train = new_x_train.groupby([race_bins, grade_bins]).apply(
            lambda x: x.sample(min_count)).reset_index(drop=True)

        # Remove some from the privileged group
        balanced_x_train = remove_instances_2(balanced_x_train, [
                                              balanced_x_train['Gender'] == 1, balanced_x_train['Race'] == 1], 3)
        global train_size_eq
        train_size_eq += len(balanced_x_train)
        return balanced_x_train.drop(h.predicted_attr, axis=1), balanced_x_train[h.predicted_attr]

    print("==========Equal Balance==========")

    h = GliomaDataset()
    h.dropper = True
    h.perturbe = perturbe
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
latex_table = df.style.to_latex(
    caption='Performance results for the Glioma Dataset',  position='p')

h = GliomaDataset()
with open(f"results/{type(h).__name__}/performance_results.tex", "w") as f:
    f.write(latex_table)
