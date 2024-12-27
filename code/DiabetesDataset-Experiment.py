import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
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

def high_imbalance():
    def perturbe(X_train, y_train):
        new_x_train = X_train.reset_index()
        new_x_train[h.predicted_attr] = y_train.reset_index()[h.predicted_attr]
        # remove 80% of females with positive outcome
        new_x_train = remove_instances_2(new_x_train, [new_x_train['Sex'] == 0, new_x_train['Diabetes_binary'] == 0], 0.80)
        new_x_train = remove_instances_2(new_x_train, [new_x_train['Sex'] == 0, new_x_train['Diabetes_binary'] == 1], 0.20)

        new_x_train = remove_instances_2(new_x_train, [
                                         new_x_train['HighBP'] == 1, new_x_train['Diabetes_binary'] == 0, new_x_train['Sex'] == 0], 0.80)
        
        new_y_train = new_x_train[h.predicted_attr]
        new_x_train = new_x_train.drop(h.predicted_attr, axis=1)
        new_x_train = new_x_train.drop('index', axis=1)

        return new_x_train, new_y_train

    print("==========High Imbalance==========")
    print("Remove 90%/ of women with positive output and 30% with positive output, respectively")
    h = DiabetesDataset()
    h.dropper = True
    h.perturbe = perturbe
    acc, f1 = h.execute_models()
    global all_acs
    all_acs += acc
    global all_f1s
    all_f1s += f1

    gen_graph_for_sets(h, "high-imbalance")

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

def check_best_values(i,j,k):
    def perturbe(X_train, y_train):
        new_x_train = X_train.reset_index()
        new_x_train[h.predicted_attr] = y_train.reset_index()[h.predicted_attr]
        # remove 80% of females with positive outcome
        new_x_train = remove_instances_2(new_x_train, [new_x_train['Sex'] == 0, new_x_train['Diabetes_binary'] == 0], i)
        new_x_train = remove_instances_2(new_x_train, [new_x_train['Sex'] == 0, new_x_train['Diabetes_binary'] == 1], j)

        new_x_train = remove_instances_2(new_x_train, [
                                            new_x_train['HighBP'] == 1, new_x_train['Diabetes_binary'] == 0, new_x_train['Sex'] == 0], k)
        
        new_y_train = new_x_train[h.predicted_attr]
        new_x_train = new_x_train.drop(h.predicted_attr, axis=1)
        new_x_train = new_x_train.drop('index', axis=1)

        return new_x_train, new_y_train

    h = DiabetesDataset()
    h.dropper = True
    h.perturbe = perturbe
    y = h.dataset[h.predicted_attr]
    x = h.dataset.drop(h.predicted_attr, axis=1)

    h.X_train, h.X_test, h.y_train, h.y_test = train_test_split(
        x, y, test_size=0.20, random_state=42
    )

    if h.dropper:
        h.X_train, h.y_train = h.perturbe(h.X_train, h.y_train)

    # join X_train and y_train
    full_dataset = h.X_train
    full_dataset[h.predicted_attr] = h.y_train
    d = h.get_metrics(full_dataset, print_metrics=False)
    return d

all_acs = []
all_f1s = []
best_cddl_sex = 0
best_cddl_age = 0
best_cddl_education = 0
best_cddl_income = 0
best_kl_sex = 0
best_kl_age = 0
best_kl_education = 0
best_kl_income = 0
best_ks_sex = 0
best_ks_age = 0
best_ks_education = 0
best_ks_income = 0
for i in np.linspace(0.0, 0.95, num=10):
    for j in np.linspace(0.0, 0.95, num=10):
         for k in np.linspace(0.0, 0.95, num=10):
            d = check_best_values(i, j, k)
            if d['CDDL (Sex, HighBP)'] > best_cddl_sex:
                best_cddl_sex = d['CDDL (Sex, HighBP)']	
            if d['CDDL (Age, HighBP)'] > best_cddl_age:
                best_cddl_age = d['CDDL (Age, HighBP)']
            if d['CDDL (Education, HighBP)'] > best_cddl_education:
                best_cddl_education = d['CDDL (Education, HighBP)']
            if d['CDDL (Income, HighBP)'] > best_cddl_income:
                best_cddl_income = d['CDDL (Income, HighBP)']

            if d['KL Divergence (Education)'] > best_kl_education:
                best_kl_education = d['KL Divergence (Education)']
            if d['KL Divergence (Income)'] > best_kl_income:
                best_kl_income = d['KL Divergence (Income)']
            if d['KL Divergence (Sex)'] > best_kl_sex:
                best_kl_sex = d['KL Divergence (Sex)']  
            if d['KL Divergence (Age)'] > best_kl_age:
                best_kl_age = d['KL Divergence (Age)']

            if d['KS (Education)'] > best_ks_education:
                best_ks_education = d['KS (Education)']
            if d['KS (Income)'] > best_ks_income:
                best_ks_income = d['KS (Income)']
            if d['KS (Sex)'] > best_ks_sex:
                best_ks_sex = d['KS (Sex)']
            if d['KS (Age)'] > best_ks_age:
                best_ks_age = d['KS (Age)']

print("BEST KL VALUE")
print(best_kl_education)
print(best_kl_income)
print(best_kl_age)
print(best_kl_sex)

print("BEST KS VALUE")
print(best_ks_education)
print(best_ks_income)
print(best_ks_age)
print(best_ks_sex)

print("BEST CDDL VALUE")
print(best_cddl_education)
print(best_cddl_income)
print(best_cddl_age)
print(best_cddl_sex)