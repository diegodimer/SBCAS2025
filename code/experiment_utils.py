import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import colors, gridspec
from matplotlib import pyplot as plt


def feature_importante(name, h):
    gs = gridspec.GridSpec(1, 3)
    fig = plt.figure(figsize=(20, 6))
    plt.subplots_adjust(bottom=0.3)

    feature_names = list(h.X_train.columns)
    # forest.feature_importances_
    importances = np.mean(
        [tree.feature_importances_ for tree in h.estimators["RandomForestClassifier"]],
        axis=0,
    )
    std = np.std(
        [tree.feature_importances_ for tree in h.estimators["RandomForestClassifier"]],
        axis=0,
    )
    forest_importances = pd.DataFrame(importances, index=feature_names)
    forest_importances["std"] = std
    forest_importances = forest_importances.sort_values(by=0, ascending=False)
    ax = fig.add_subplot(gs[0])
    forest_importances[0].plot.bar(ax=ax, yerr=forest_importances["std"])
    ax.set_title("Feature Importance RandomForestClassifier")

    feature_names = list(h.X_train.columns)
    importances = np.mean(
        [tree.feature_importances_ for tree in h.estimators["DecisionTreeClassifier"]],
        axis=0,
    )
    std = np.std(
        [tree.feature_importances_ for tree in h.estimators["DecisionTreeClassifier"]],
        axis=0,
    )
    forest_importances = pd.DataFrame(importances, index=feature_names)
    forest_importances["std"] = std
    forest_importances = forest_importances.sort_values(by=0, ascending=False)
    ax = fig.add_subplot(gs[1])
    forest_importances[0].plot.bar(ax=ax, yerr=forest_importances["std"])
    ax.set_title("Feature Importance DecisionTreeClassifier")

    importances = np.mean(
        [pow(math.e, w.coef_[0]) for w in h.estimators["LogisticRegression"]], axis=0
    )  # pow(math.e, w)
    std = np.std(
        [pow(math.e, w.coef_[0]) for w in h.estimators["LogisticRegression"]], axis=0
    )
    logreg_importances = pd.DataFrame(importances, index=feature_names)
    logreg_importances["std"] = std
    logreg_importances = logreg_importances.sort_values(by=0, ascending=False)
    ax = fig.add_subplot(gs[2])
    logreg_importances[0].plot.bar(ax=ax, yerr=std)
    ax.set_title("Feature Importance LogisticRegression")
    plt.tight_layout()
    Path(f"results/{type(h).__name__}/{name}").mkdir(exist_ok=True, parents=True)
    fig.savefig(
        f"results/{type(h).__name__}/{name}/importances.png".replace(">", ""),
    )
    plt.close(fig)


def generate_model_bars(h, name, model_dic, bar_name, use_percentage_values=True):
    gs = gridspec.GridSpec(1, 3)
    fig = plt.figure(figsize=(20,5))
    for idx, model in enumerate(h.models):
        ax = fig.add_subplot(gs[idx])
        data = model_dic[model]
        labels = list(data.keys())
        values = list(data.values())
        percents = [100.0 * i / sum(values) for i in values]

        bars = ax.bar(labels, percents if use_percentage_values else values, color=plt.cm.Paired(np.arange(len(values))))
        ax.set_title(f"{model}")
        ax.set_ybound(0, 100)
        ax.set_ylabel("Count")
        ax.set_xlabel("Categories")
        ax.get_xaxis().set_visible(False)  # remove text from x axis
        plt.legend(bars, labels, loc="best")

        for bar, percent, val in zip(bars, percents, list(model_dic[model].values())):
            height = bar.get_height()
            ax.annotate(
                f"{percent:.2f}% ({val})" if use_percentage_values else f"{val:.2f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    Path(f"results/{type(h).__name__}/{name}").mkdir(exist_ok=True, parents=True)
    fig.savefig(f"results/{type(h).__name__}/{name}/{bar_name}.png".replace(">", ""))
    plt.close(fig)


def remove_instances(x, conditions, value):
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


def generate_charts(h, name, full_dataset_test):
    avg_acc = defaultdict(list)
    d = defaultdict(lambda: defaultdict(dict))
    d_wrongs = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    d_correct = defaultdict(lambda: defaultdict(dict))
    d_relative = defaultdict(lambda: defaultdict(dict))
    for attr in h.protected_attr_mappings.keys():
        for val in h.protected_attr_mappings[attr].keys():
            for model in h.models:
                d[attr][model][f"{val} predicted correctly"] = len(
                    full_dataset_test.loc[
                        (
                            full_dataset_test[attr].isin(h.protected_attr_mappings[attr][val])
               
                        )
                        & (
                            full_dataset_test[model].isin(full_dataset_test[h.predicted_attr])
                        )
                    ]
                )
                d[attr][model][f"{val} false positive"] = len(
                    full_dataset_test.loc[
                        (
                            full_dataset_test[attr].isin(h.protected_attr_mappings[attr][val])
                        )
                        & (full_dataset_test[h.predicted_attr] == 0)
                        & (full_dataset_test[model] == 1)
                    ]
                )
                d[attr][model][f"{val} false negative"] = len(
                    full_dataset_test.loc[
                        (
                            full_dataset_test[attr].isin(h.protected_attr_mappings[attr][val])
                        )
                        & (full_dataset_test[h.predicted_attr] == 1)
                        & (full_dataset_test[model] == 0)
                    ]
                )
                d_wrongs[attr][val][model][f"{val} predicted correctly"] = len(
                    full_dataset_test.loc[
                        (
                            full_dataset_test[attr].isin(h.protected_attr_mappings[attr][val])
                        )
                        & (
                            full_dataset_test[h.predicted_attr]
                            == full_dataset_test[model]
                        )
                    ]
                )
                d_wrongs[attr][val][model][f"{val} false negative"] = len(
                    full_dataset_test.loc[
                        (
                            full_dataset_test[attr].isin(h.protected_attr_mappings[attr][val])
                        )
                        & (full_dataset_test[h.predicted_attr] == 1)
                        & (full_dataset_test[model] == 0)
                    ]
                )
                d_wrongs[attr][val][model][f"{val} false positive"] = len(
                    full_dataset_test.loc[
                        (
                            full_dataset_test[attr].isin(h.protected_attr_mappings[attr][val])
                        )
                        & (full_dataset_test[h.predicted_attr] == 0)
                        & (full_dataset_test[model] == 1)
                    ]
                )
                d_correct[attr][model][f"{val} predicted correctly"] = len(
                    full_dataset_test.loc[
                        (
                            full_dataset_test[attr].isin(h.protected_attr_mappings[attr][val])
                        )
                        & (
                            full_dataset_test[model]
                            == full_dataset_test[h.predicted_attr]
                        )
                    ]
                )
                avg_acc[val].append(
                    (d_correct[attr][model][f"{val} predicted correctly"])
                    / (
                        len(
                            full_dataset_test.loc[
                                (
                                    full_dataset_test[attr].isin(h.protected_attr_mappings[attr][val])
                                )
                            ]
                        )
                    )
                )
                
                d_relative[attr][model].update({
                    f"{val} predicted correctly" : 100 * ((d[attr][model][f"{val} predicted correctly"]) / 
                                                        (d[attr][model][f"{val} predicted correctly"] + d[attr][model][f"{val} false positive"] + d[attr][model][f"{val} false negative"])),
                    f"{val} predicted wrongly": 100 * ( (d[attr][model][f'{val} false positive'] + d[attr][model][f'{val} false negative']) / 
                                                        (d[attr][model][f"{val} predicted correctly"] + d[attr][model][f"{val} false positive"] + d[attr][model][f"{val} false negative"]))
                })
        
    for attr in d.keys():
        generate_model_bars(h, name, d[attr], f"barchart-complete-{attr}")

    for attr in d_wrongs.keys():
        for val in d_wrongs[attr].keys():
            generate_model_bars(h, name, d_wrongs[attr][val], f"barchart-{attr}-{val}")

    for attr in d_correct.keys():
        generate_model_bars(h, name, d_correct[attr], f"barchart-correct-{attr}")

    for attr in d_relative.keys():
            # order d_relative[attr] by value
            for key in d_relative[attr].keys():
                d_relative[attr][key] = dict(sorted(d_relative[attr][key].items(), key=lambda x:x[1], reverse=True))

            generate_model_bars(h, name, d_relative[attr], f"barchart-relative-{attr}", use_percentage_values=False)
    
    avg_acc_df = {}
    for key in avg_acc:
        avg_acc_df[key] = round(((sum(avg_acc[key]))/ (len(avg_acc[key]))*100 ),3)
    df = pd.DataFrame.from_dict(avg_acc_df, orient='index', columns=['Mean Accuracy'])
    print(df)
    with open(f"results/{type(h).__name__}/{name}/mean_accuracy.txt".replace(">", ""), "w") as f:
        f.write(df.style.to_latex(caption='Mean accuracy for each protected attribute',  position='p'))

def evaluate_train_and_test_sets(h, name, stratify_age=False):
    gs_test = {
        attr: gridspec.GridSpec(round(h.num_repetitions / 2), 2)
        for attr in h.protected_attr
    }
    gs_train = {
        attr: gridspec.GridSpec(round(h.num_repetitions / 2), 2)
        for attr in h.protected_attr
    }
    fig_testsets = {attr: (plt.figure(figsize=(10, 20))) for attr in h.protected_attr}
    fig_trainsets = {attr: (plt.figure(figsize=(10, 20))) for attr in h.protected_attr}

    for attr in h.protected_attr:
        for i in range(h.num_repetitions):
            train_set = h.x_train_list[i].reset_index()
            train_set[h.predicted_attr] = h.y_train_list[i].reset_index()[
                h.predicted_attr
            ]
            test_set = h.x_test_list[i].reset_index()
            test_set[h.predicted_attr] = h.y_test_list[i].reset_index()[
                h.predicted_attr
            ]
            for model in h.models:
                y_hats = pd.DataFrame(h.predicted_list[model][i])
                test_set[model] = y_hats.reset_index()[0]

            if stratify_age:
                h.stratify_age(test_set)
                h.stratify_age(train_set)

            ax = fig_testsets[attr].add_subplot(gs_test[attr][i])
            h.gen_graph(
                dataset=test_set,
                file_name=f"{name}/{i}-test",
                graph_title=f"Test set #{i}",
                labels_labels=h.protected_attr_mappings[attr].keys(),
                ax=ax,
                protected_attr=attr,
            )

            ax2 = fig_trainsets[attr].add_subplot(gs_train[attr][i])
            h.gen_graph(
                dataset=train_set,
                file_name=f"{name}/{i}-test",
                graph_title=f"Train set #{i}",
                labels_labels=h.protected_attr_mappings[attr].keys(),
                ax=ax2,
                protected_attr=attr,
            )
        Path(f"results/{type(h).__name__}/{name}").mkdir(exist_ok=True, parents=True)
        fig_testsets[attr].savefig(
            f"results/{type(h).__name__}/{name}/testSetsGrouped-{attr}.png".replace(">", "")
        )
        fig_trainsets[attr].savefig(
            f"results/{type(h).__name__}/{name}/trainSetsGrouped-{attr}.png".replace(">", "")
        )


def get_full_sets_graphs(h, name, stratify_age=False):
    full_dataset_train = pd.concat(
        [h.x_train_list[i].reset_index() for i in range(h.num_repetitions)]
    )
    full_dataset_train[h.predicted_attr] = pd.concat(
        [
            h.y_train_list[i].reset_index()[h.predicted_attr]
            for i in range(h.num_repetitions)
        ]
    )
    full_dataset_train.drop("index", axis=1)
    plt.rcParams["figure.autolayout"] = True
    full_dataset_test = pd.concat(
        [h.x_test_list[i].reset_index() for i in range(h.num_repetitions)]
    )
    full_dataset_test[h.predicted_attr] = pd.concat(
        [
            h.y_test_list[i].reset_index()[h.predicted_attr]
            for i in range(h.num_repetitions)
        ]
    )
    full_dataset_test.drop("index", axis=1)

    # ADD MODELS PREDICTIONS TO FULL DATASET TRAIN
    for model in h.models:
        df_hats = pd.concat(
            [
                pd.DataFrame(h.predicted_list[model][i]).reset_index()[0]
                for i in range(h.num_repetitions)
            ]
        )
        full_dataset_test[model] = df_hats

    # PRINT METRICS (IT IS OVER ALL TRAINING DATA)
    print(
        f"\nMetrics calculated over the train datasets (concatenanted from all {h.num_repetitions} repetitions)"
    )
    d = h.get_metrics(full_dataset_train)
    # make all values with 3 decimal places
    d = {k: round(v, 3) for k, v in d.items()}
    df = pd.DataFrame(d.items(), columns = ['Metric', 'Value'])

    with open(f"results/{type(h).__name__}/{name}/metrics_train.txt".replace(">", ""), "w") as f:
        f.write(df.style.hide(axis='index').to_latex( caption=f'Metrics calculated over the train datasets (from {h.num_repetitions} repetitions)',  position='p'))

    if stratify_age:
        h.stratify_age(full_dataset_test)

    fig, ax = plt.subplots()
    total_size = len(full_dataset_test)
    right = []
    wrong = []
    labels = list(h.models)
    for p in labels:
        right.append(
            (
                len(
                    full_dataset_test.loc[
                        (full_dataset_test[p] == full_dataset_test[h.predicted_attr])
                    ]
                )
                / total_size
            )
            * 100.0
        )
        wrong.append(
            (
                len(
                    full_dataset_test.loc[
                        (full_dataset_test[p] != full_dataset_test[h.predicted_attr])
                    ]
                )
                / total_size
            )
            * 100.0
        )

    df = pd.DataFrame({"correct": right, "wrong": wrong}, index=labels)
    ax = df.plot.barh(ax=ax, color=["green", "red"])
    ax.legend()
    for bars in ax.containers:  # if the bars should have the values
        plt.bar_label(
            bars, labels=[f"{x:,.2f}%" for x in bars.datavalues], label_type="center"
        )
    Path(f"results/{type(h).__name__}/{name}").mkdir(exist_ok=True, parents=True)
    fig.savefig(f"results/{type(h).__name__}/{name}/FullTest-Predictions.png".replace(">", ""))
    plt.close(fig)

    # GENERATE GRAPHS FOR FULL TRAINING AND TEST DATA
    for attr in h.protected_attr:
        h.gen_graph(
            attr,
            df_type=f"{name}/FulltrainDataset-{attr}",
            dataset=full_dataset_train,
            labels_labels=list(h.protected_attr_mappings[attr].keys()),
            graph_title="Complete train set",
        )
        h.gen_graph(
            attr,
            df_type=f"{name}/FulltestDataset-{attr}",
            dataset=full_dataset_test,
            labels_labels=list(h.protected_attr_mappings[attr].keys()),
            graph_title="Complete test set",
        )

    return full_dataset_test
