import itertools
from collections import defaultdict
from pathlib import Path

import graphviz
import matplotlib.pyplot as plt
import numpy as np
from PreTrainingBias.PreTrainingBias import PreTrainingBias
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from ydata_profiling import ProfileReport


class BaseDataset:
    """
    BaseDataset is the base class to be extended for new datasets.

    Methods
    -------
    __init__():
        Initializes the BaseDataset class with default values for various attributes.
    _init_models(random_state):
        Initializes the machine learning models with the given random state.
    execute_models():
        Executes the models, generates training and testing sets, and calculates accuracy and F1 scores.
    _gen_train_test_sets(random_state):
        Generates training and testing sets based on the given random state.
    perturbe(x_train):
        Placeholder method for perturbing the training data.
    _gen_pp_report():
        Generates a profile report of the dataset.
    _run_model(model):
        Runs the given model and returns accuracy and F1 scores.
    evaluate_metrics(protected_attribute, privileged_group, group_variable, dataset=None, cddl_only=False, print_metrics=True):
        Evaluates and prints various metrics for the given protected attribute and group.
    gen_graph(protected_attr=None, labels_labels=None, outcomes_labels=None, dataset=None, predicted_attr=None, file_name=None, df_type=None, graph_title=None, ax=None):
        Generates and saves a graph based on the given parameters.
    save_tree():
        Saves a visualization of the decision tree model.
    num_models():
        Returns the number of models initialized.

    """

    def __init__(self) -> None:
        """
        Initialize the BaseDataset class with default values for various attributes.

        Attributes
        ----------
        dataset (pandas.DataFrame): The dataset in pandas dataframe format.
        type_schema (dict): The schema for the dataset's PP report.
        predicted_attr (str): The column name of the predicted attribute.
        max_iter (int): Maximum number of iterations for LogisticRegression.
        n_estimators (int): Number of estimators for RandomForest.
        random_state (int): Random state used for all models and split.
        max_depth (int): Maximum depth for DecisionTree and RandomForest.
        criterion (str): Criterion used for DecisionTree and RandomForest.
        positive_outcome (str): The positive outcome of the predicted attribute.
        negative_outcome (str): The negative outcome of the predicted attribute.
        num_repetitions (int): Number of repetitions to run the models.
        protected_attr (list): List of protected attributes.
        protected_attr_mappings (dict): Dictionary of mappings for the protected attributes.
        x_train_list (list): List to store training features.
        x_test_list (list): List to store testing features.
        y_train_list (list): List to store training labels.
        y_test_list (list): List to store testing labels.
        predicted_list (defaultdict): Default dictionary to store predicted values.
        model_conf_matrix (defaultdict): Default dictionary to store model confusion matrices.
        accs (defaultdict): Default dictionary to store accuracy scores.
        f1s (defaultdict): Default dictionary to store F1 scores.
        models (defaultdict): Default dictionary to store models.
        estimators (defaultdict): Default dictionary to store estimators.
        ptb (PreTrainingBias): Instance of PreTrainingBias class.
        dropper (bool): Flag to indicate if the perturbe function should be called.

        """
        self.dataset = None # dataset in pandas dataframe format
        self.type_schema = None # schema for the dataset's PP report
        self.predicted_attr = None # the column name of the predicted attribute
        self.max_iter = None # used for LogisticRegression
        self.n_estimators = None # used for RandomForest
        self.random_state = 42 # used for all models and split
        self.max_depth = None # used for DecisionTree and RandomForest
        self.criterion = None # used for DecisionTree and RandomForest
        self.positive_outcome = None # the positive outcome of the predicted attribute
        self.negative_outcome = None # the negative outcome of the predicted attribute
        self.num_repetitions = None # number of repetitions to run the models
        self.protected_attr = None # list of protected attributes
        self.protected_attr_mappings = None # dictionary of mappings for the protected attributes

        # variables to be used by this main class
        self.x_train_list = []
        self.x_test_list = []
        self.y_train_list = []
        self.y_test_list = []
        self.predicted_list = defaultdict(list)
        self.model_conf_matrix = defaultdict(list)
        self.accs = defaultdict(list)
        self.f1s = defaultdict(list)
        self.models = defaultdict()
        self.estimators = defaultdict(list)
        self.ptb = PreTrainingBias()
        self.dropper = False
        np.random.seed(42)

    def _init_models(self, random_state):
        self.models["LogisticRegression"] = LogisticRegression(
            max_iter=self.max_iter, random_state=random_state
        )
        self.models["DecisionTreeClassifier"] = DecisionTreeClassifier(
            criterion="entropy", random_state=random_state, max_depth=self.max_depth
        )
        self.models["RandomForestClassifier"] = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=random_state,
            max_depth=self.max_depth,
        )

    def execute_models(self):
        self._gen_pp_report()
        self._init_models(self.random_state)

        for i in range(1, self.num_repetitions + 1):
            self._gen_train_test_sets(i)
            for model_name in self.models:
                acc, f1 = self._run_model(self.models[model_name])
                self.accs[model_name].append(acc)
                self.f1s[model_name].append(f1)
        acc_return = {}
        f1_return = {}
        for model_name in self.accs:
            acc = self.accs[model_name]
            f1 = self.f1s[model_name]
            acc_mean = round(sum(acc) / len(acc), 3)
            f1_mean = round(sum(f1) / len(f1), 3)
            # print("{: <30}{: >30.3f}".format(model_name + " acc", acc_mean))
            # print("{: <30}{: >30.3f}".format(model_name + " f1", f1_mean))
            acc_return[model_name] = acc_mean
            f1_return[model_name] = f1_mean
        return acc_return, f1_return

    def _gen_train_test_sets(self, random_state):
        y = self.dataset[self.predicted_attr]
        x = self.dataset.drop(self.predicted_attr, axis=1)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=0.20, random_state=random_state
        )

        if self.dropper:
            self.X_train, self.y_train = self.perturbe(self.X_train, self.y_train)

        self.x_train_list.append(self.X_train)
        self.x_test_list.append(self.X_test)
        self.y_train_list.append(self.y_train)
        self.y_test_list.append(self.y_test)

    def perturbe(self, x_train):
        raise ("You need to implement the perturbe method")

    def _gen_pp_report(self):
        my_file = Path(f"results/{type(self).__name__}/report.html")
        if not my_file.exists():
            Path(f"results/{type(self).__name__}").mkdir(parents=True, exist_ok=True)
            self.dataset.profile_report(type_schema=self.type_schema).to_file(f"results/{type(self).__name__}/report.html")
            # ProfileReport(self.dataset).to_file(f"results/{type(self).__name__}/report.html")

    def _run_model(self, model):
        model_name = type(model).__name__
        model.fit(self.X_train, self.y_train)
        if type(model).__name__ == "RandomForestClassifier":
            for i in model.estimators_:
                self.estimators["RandomForestClassifier"].append(i)
        else:
            self.estimators[type(model).__name__].append(model)

        model_predicted = model.predict(self.X_test)
        self.predicted_list[model_name].append(model_predicted)
        self.model_conf_matrix[model_name].append(
            confusion_matrix(self.y_test, model_predicted)
        )
        model_acc_score = accuracy_score(self.y_test, model_predicted) * 100
        model_f1_score = f1_score(self.y_test, model_predicted) * 100
        return model_acc_score, model_f1_score

    def evaluate_metrics(
        self,
        protected_attribute,
        privileged_group,
        group_variable,
        dataset=None,
        cddl_only=False,
        print_metrics=True,
    ):
        dataset = self.dataset if dataset is None else dataset

        dic = self.ptb.global_evaluation(
            dataset,
            self.predicted_attr,
            self.positive_outcome,
            protected_attribute,
            privileged_group,
            group_variable,
        )
        out_dic = {}
        for key in dic:
            if cddl_only:
                if "CDDL" in key:
                    if print_metrics:
                        print(f"{key: <30}{dic[key]: >30.3f}")
                    out_dic[key] = dic[key]
                    break
            else:
                if print_metrics:
                    print(f"{key: <30}{dic[key]: >30.3f}")
                out_dic[key] = dic[key]
        return out_dic

    def gen_graph(
        self,
        protected_attr=None,
        labels_labels=None,
        outcomes_labels=None,
        dataset=None,
        predicted_attr=None,
        file_name=None,
        df_type=None,
        graph_title=None,
        ax=None,
    ):
        dataset = self.dataset if dataset is None else dataset
        predicted_attr = (
            self.predicted_attr if predicted_attr is None else predicted_attr
        )
        protected_attr = (
            self.protected_attr if protected_attr is None else protected_attr
        )
        protected_attr = (
            [protected_attr] if type(protected_attr) is str else protected_attr
        )

        for attr in protected_attr:
            labels =  list(self.protected_attr_mappings[attr].values())#dataset[attr].unique().tolist()
            outcomes = dataset[predicted_attr].unique().tolist()
            outcomes.sort()
            bar_ind = []
            bar_list = []
            for i in outcomes:
                for j in labels:
                    bar_ind.append(
                        len(
                            dataset[
                                (dataset[predicted_attr] == i) & (dataset[attr].isin(j))
                            ]
                        )
                    )
                bar_list.append(bar_ind)
                bar_ind = []

            # the width of the bars: can also be len(x) sequence
            width = 0.35
            fig = None
            if ax is None:
                fig, ax = plt.subplots()
                plt.figure(figsize=(40, 24))

            previous = None

            for i, j in enumerate(bar_list):
                if i == 0:
                    ax.bar(list(range(len(j))), j, width, label=f"{i}")
                    previous = np.array(j)
                if i != 0:
                    ax.bar(list(range(len(j))), j, width, label=f"{i}", bottom=previous)
                    previous += np.array(j)

            if labels_labels is not None:
                x_ticks_labels = labels_labels
                ax.set_xticks(list(range(len(labels_labels))))
                ax.set_xticklabels(x_ticks_labels)

            ax.legend(outcomes_labels) if outcomes_labels is not None else ax.legend(
                title=predicted_attr,
                loc='best'
            )

            if graph_title is not None:
                ax.set_title(graph_title)

            ax.set_ylabel("count")
            for bars in ax.containers:  # if the bars should have the values
                ax.bar_label(bars)
            
            path_dir = Path(f"results/{type(self).__name__}")
            if not path_dir.exists():
                path_dir.mkdir(parents=True, exist_ok=True)
            
            if fig is not None:
                fig.savefig(
                    f"results/{type(self).__name__}/{df_type}-{predicted_attr}-{attr}.png".replace(
                        ">", ""
                    )
                ) if file_name is None else fig.savefig(
                    f"results/{type(self).__name__}/{file_name}.png".replace(">", "")
                )
                plt.close(fig)

    def save_tree(self):
        dot_data = tree.export_graphviz(
            self.models["Decision Tree"],
            out_file=None,
            feature_names=self.X_train.columns,
            class_names=[str(x) for x in self.y_test.unique()],
            filled=True,
            rounded=True,
            special_characters=True,
            impurity=False,
            max_depth=3,
        )

        graph = graphviz.Source(dot_data)
        graph.render(f"tree-results/{type(self).__name__}")

    def num_models(self):
        return len(self.models)

    def gen_var_dist(self, protected_attr: str, predicted_attr: str, labels: list[str]=None, variation_name: str=""):
        Positive = []
        Negative = []
        for i in sorted(self.dataset[protected_attr].unique()):
            Positive.append(len(self.dataset[(self.dataset[protected_attr] == i) & (self.dataset[predicted_attr] == self.positive_outcome)]))
            Negative.append(len(self.dataset[(self.dataset[protected_attr] == i) & (self.dataset[predicted_attr] == self.negative_outcome)]))
        x = np.arange(len(labels))
        width = 0.30  # the width of the bars

        fig, ax = plt.subplots()

        plt.xticks(rotation=90)
        ax.bar(x - width/2, Positive, width, label='Positive')
        ax.bar(x + width/2, Negative, width, label='Negative')

        ax.set_ylabel('Values')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        # rotate x label
        ax.set_title(f"Dist. of {predicted_attr} by {protected_attr}")
        ax.legend(loc='best')
        plt.tight_layout()

        path_dir = Path(f"results/{type(self).__name__}/{variation_name}/")
        if not path_dir.exists():
            path_dir.mkdir(parents=True, exist_ok=True)

        fig.savefig(f"results/{type(self).__name__}/{variation_name}/dist-{protected_attr}-{predicted_attr}")