import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Datasets.DiabetesDataset import DiabetesDataset
from experiment_utils import (evaluate_train_and_test_sets, feature_importante,
                              generate_charts, get_full_sets_graphs)
from tabulate import tabulate


def remove_instances_2(x, conditions: list, percentage: float):
    new_x = x.loc[np.logical_and.reduce(conditions)]
    new_x_size = len(new_x)
    drop_indices = np.random.choice(
        new_x.index, min(percentage, new_x_size) if percentage >= 1
        else min(round(new_x_size*percentage), new_x_size), replace=False)
    new_xtrain = x.drop(drop_indices)
    return new_xtrain


def gen_graph_for_sets(h: DiabetesDataset, name: str):
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

    h = DiabetesDataset()
    print("==========Original Dataset===========")
    h.perturbe = perturbe
    h.dropper = True
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

    def perturbe(X_train, y_train):
        new_x_train = X_train.reset_index()
        new_x_train[h.predicted_attr] = y_train.reset_index()[h.predicted_attr]
        new_x_train = remove_instances_2(
            new_x_train, [new_x_train['Sex'] == 0, new_x_train['Diabetes_binary'] == 0], .85)
        new_x_train = remove_instances_2(
            new_x_train, [new_x_train['Sex'] == 1, new_x_train['Diabetes_binary'] == 1], 0.85)

        new_x_train = remove_instances_2(
            new_x_train, [new_x_train['Age'] != 1, new_x_train['Diabetes_binary'] == 0], .55)
        new_x_train = remove_instances_2(
            new_x_train, [new_x_train['Age'] == 1, new_x_train['Diabetes_binary'] == 1], .45)

        new_x_train = remove_instances_2(new_x_train, [
            new_x_train['HighBP'] == 0, new_x_train['Diabetes_binary'] == 0, new_x_train['Sex'] == 0], 0.2)
        new_x_train = remove_instances_2(new_x_train, [
            new_x_train['HighBP'] == 1, new_x_train['Diabetes_binary'] == 1, new_x_train['Sex'] == 1], 0.2)

        new_y_train = new_x_train[h.predicted_attr]
        new_x_train = new_x_train.drop(h.predicted_attr, axis=1)
        new_x_train = new_x_train.drop('index', axis=1)

        global train_size_hi
        train_size_hi += len(new_x_train)
        return new_x_train, new_y_train

    print("==========High Imbalance==========")
    print(
        """
        Remove 85% of women with negative output and 85% of men with positive output, respectively\n"
        Remove 55% of people with age between 18 and 24 with positive output and 45%
        of people with age above 24 with negative output, respectively\n"
        Remove 20% of women with low blood pressure with negative output and 20%
        of men with high blood pressure with positive output, respectively\n"
        """
    )
    h = DiabetesDataset()
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

    def perturbe(X_train, y_train):
        new_x_train = X_train.reset_index()
        new_x_train[h.predicted_attr] = y_train.reset_index()[h.predicted_attr]

        unprivileged_group = [1]

        # Separate the dataset into privileged and unprivileged groups
        privileged_df = new_x_train[~new_x_train['Age'].isin(
            unprivileged_group)]
        unprivileged_df = new_x_train[new_x_train['Age'].isin(
            unprivileged_group)]

        # Further separate by HighBP and Sex
        privileged_highbp_male = privileged_df[(
            privileged_df['HighBP'] == 1) & (privileged_df['Sex'] == 1)]
        privileged_highbp_female = privileged_df[(
            privileged_df['HighBP'] == 1) & (privileged_df['Sex'] == 0)]
        unprivileged_highbp_male = unprivileged_df[(
            unprivileged_df['HighBP'] == 1) & (unprivileged_df['Sex'] == 1)]
        unprivileged_highbp_female = unprivileged_df[(
            unprivileged_df['HighBP'] == 1) & (unprivileged_df['Sex'] == 0)]

        privileged_lowbp_male = privileged_df[(
            privileged_df['HighBP'] == 0) & (privileged_df['Sex'] == 1)]
        privileged_lowbp_female = privileged_df[(
            privileged_df['HighBP'] == 0) & (privileged_df['Sex'] == 0)]
        unprivileged_lowbp_male = unprivileged_df[(
            unprivileged_df['HighBP'] == 0) & (unprivileged_df['Sex'] == 1)]
        unprivileged_lowbp_female = unprivileged_df[(
            unprivileged_df['HighBP'] == 0) & (unprivileged_df['Sex'] == 0)]

        # Get the minimum count for balancing
        min_count = min(len(privileged_highbp_male), len(privileged_highbp_female),
                        len(unprivileged_highbp_male),
                        len(unprivileged_highbp_female),
                        len(privileged_lowbp_male),
                        len(privileged_lowbp_female),
                        len(unprivileged_lowbp_male),
                        len(unprivileged_lowbp_female))

        # Sample min_count instances from each group
        balanced_privileged_highbp_male = privileged_highbp_male.sample(
            min_count)
        balanced_privileged_highbp_female = privileged_highbp_female.sample(
            min_count)
        balanced_unprivileged_highbp_male = unprivileged_highbp_male.sample(
            min_count)
        balanced_unprivileged_highbp_female = unprivileged_highbp_female.sample(
            min_count)
        balanced_privileged_lowbp_male = privileged_lowbp_male.sample(
            min_count)
        balanced_privileged_lowbp_female = privileged_lowbp_female.sample(
            min_count)
        balanced_unprivileged_lowbp_male = unprivileged_lowbp_male.sample(
            min_count)
        balanced_unprivileged_lowbp_female = unprivileged_lowbp_female.sample(
            min_count)

        # Combine the balanced datasets
        balanced_x_train = (pd.concat([balanced_privileged_highbp_male, balanced_privileged_highbp_female,
                                      balanced_unprivileged_highbp_male, balanced_unprivileged_highbp_female,
                                      balanced_privileged_lowbp_male, balanced_privileged_lowbp_female,
                                      balanced_unprivileged_lowbp_male, balanced_unprivileged_lowbp_female])
                            .reset_index(drop=True))

        global train_size_eq
        train_size_eq += len(balanced_x_train)
        balanced_x_train = balanced_x_train.drop('index', axis=1)
        return balanced_x_train.drop(h.predicted_attr, axis=1), balanced_x_train[h.predicted_attr]

    print("==========Equal Balance==========")

    h = DiabetesDataset()
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
    caption='Performance results for the Diabetes Dataset',  position='p')

h = DiabetesDataset()
with open(f"results/{type(h).__name__}/performance_results.tex", "w") as f:
    f.write(latex_table)
