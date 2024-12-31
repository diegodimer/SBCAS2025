import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Datasets.DiabetesDataset import DiabetesDataset
from experiment_utils import (evaluate_train_and_test_sets, feature_importante,
                              generate_charts, get_full_sets_graphs)


def remove_instances_2(x, conditions: list, percentage: float):
    new_x = x.loc[np.logical_and.reduce(conditions)]
    new_x_size = len(new_x)
    drop_indices = np.random.choice(
        new_x.index, min(percentage, new_x_size) if percentage >= 1 else min(round(new_x_size*percentage), new_x_size), replace=False)
    new_xtrain = x.drop(drop_indices)
    return new_xtrain

def gen_graph_for_sets(h: DiabetesDataset, name: str):
    full_dataset_test = get_full_sets_graphs(h, name)
    generate_charts(h, name, full_dataset_test)
    evaluate_train_and_test_sets(h, name)

    plt.close("all")

    feature_importante(name, h)

def original_dataset():
    h = DiabetesDataset()
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
        new_x_train = remove_instances_2(new_x_train, [new_x_train['Sex'] == 0, new_x_train['Diabetes_binary'] == 0], .85)
        new_x_train = remove_instances_2(new_x_train, [new_x_train['Sex'] == 1, new_x_train['Diabetes_binary'] == 1], 0.85)

        new_x_train = remove_instances_2(new_x_train, [new_x_train['Age'] != 1, new_x_train['Diabetes_binary'] == 0], .55)
        new_x_train = remove_instances_2(new_x_train, [new_x_train['Age'] == 1, new_x_train['Diabetes_binary'] == 1], .45)

        new_x_train = remove_instances_2(new_x_train, [
                                            new_x_train['HighBP'] == 0, new_x_train['Diabetes_binary'] == 0, new_x_train['Sex'] == 0], 0.2)
        new_x_train = remove_instances_2(new_x_train, [
                                            new_x_train['HighBP'] == 1, new_x_train['Diabetes_binary'] == 1, new_x_train['Sex'] == 1], 0.2)
        
        new_y_train = new_x_train[h.predicted_attr]
        new_x_train = new_x_train.drop(h.predicted_attr, axis=1)
        new_x_train = new_x_train.drop('index', axis=1)

        print(f"New size: {len(new_x_train)}")
        return new_x_train, new_y_train

    print("==========High Imbalance==========")
    print(
        "Remove 85% of women with negative output and 85% of men with positive output, respectively\n"
        "Remove 75% of people with age between 18 and 24 with positive output and 85% of people with age above 24 with negative output, respectively\n"
        "Remove 20% of women with low blood pressure with negative output and 20% of men with high blood pressure with positive output, respectively\n"
    )
    h = DiabetesDataset()
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
        new_x_train = X_train.reset_index()
        new_x_train[h.predicted_attr] = y_train.reset_index()[h.predicted_attr]

        unprivileged_group = [1] 

        # Separate the dataset into privileged and unprivileged groups
        privileged_df = new_x_train[~new_x_train['Age'].isin(unprivileged_group)]
        unprivileged_df = new_x_train[new_x_train['Age'].isin(unprivileged_group)]

        # Further separate by HighBP and Sex
        privileged_highbp_male = privileged_df[(privileged_df['HighBP'] == 1) & (privileged_df['Sex'] == 1)]
        privileged_highbp_female = privileged_df[(privileged_df['HighBP'] == 1) & (privileged_df['Sex'] == 0)]
        unprivileged_highbp_male = unprivileged_df[(unprivileged_df['HighBP'] == 1) & (unprivileged_df['Sex'] == 1)]
        unprivileged_highbp_female = unprivileged_df[(unprivileged_df['HighBP'] == 1) & (unprivileged_df['Sex'] == 0)]

        privileged_lowbp_male = privileged_df[(privileged_df['HighBP'] == 0) & (privileged_df['Sex'] == 1)]
        privileged_lowbp_female = privileged_df[(privileged_df['HighBP'] == 0) & (privileged_df['Sex'] == 0)]
        unprivileged_lowbp_male = unprivileged_df[(unprivileged_df['HighBP'] == 0) & (unprivileged_df['Sex'] == 1)]
        unprivileged_lowbp_female = unprivileged_df[(unprivileged_df['HighBP'] == 0) & (unprivileged_df['Sex'] == 0)]

        # Get the minimum count for balancing
        min_count = min(len(privileged_highbp_male), len(privileged_highbp_female), len(unprivileged_highbp_male), len(unprivileged_highbp_female),
                        len(privileged_lowbp_male), len(privileged_lowbp_female), len(unprivileged_lowbp_male), len(unprivileged_lowbp_female))

        # Sample min_count instances from each group
        balanced_privileged_highbp_male = privileged_highbp_male.sample(min_count)
        balanced_privileged_highbp_female = privileged_highbp_female.sample(min_count)
        balanced_unprivileged_highbp_male = unprivileged_highbp_male.sample(min_count)
        balanced_unprivileged_highbp_female = unprivileged_highbp_female.sample(min_count)
        balanced_privileged_lowbp_male = privileged_lowbp_male.sample(min_count)
        balanced_privileged_lowbp_female = privileged_lowbp_female.sample(min_count)
        balanced_unprivileged_lowbp_male = unprivileged_lowbp_male.sample(min_count)
        balanced_unprivileged_lowbp_female = unprivileged_lowbp_female.sample(min_count)

        # Combine the balanced datasets
        balanced_x_train = pd.concat([balanced_privileged_highbp_male, balanced_privileged_highbp_female,
                                      balanced_unprivileged_highbp_male, balanced_unprivileged_highbp_female,
                                      balanced_privileged_lowbp_male, balanced_privileged_lowbp_female,
                                      balanced_unprivileged_lowbp_male, balanced_unprivileged_lowbp_female]).reset_index(drop=True)

        print(f"Balanced size: {len(balanced_x_train)}")
        balanced_x_train = balanced_x_train.drop('index', axis=1)
        return balanced_x_train.drop(h.predicted_attr, axis=1), balanced_x_train[h.predicted_attr]
    
    print("==========Equal Balance==========")

    h = DiabetesDataset()
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
