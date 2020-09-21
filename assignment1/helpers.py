import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import uuid

from collections import Sequence
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score

plt.style.use("seaborn-whitegrid")

def fit_model(ohe_columns, classifier, params, X_train, y_train):
    return Pipeline([
        ("preprocessor",
            ColumnTransformer([
                ("encoder", OneHotEncoder(), ohe_columns)
            ],
            remainder = 'passthrough')
        ),
        ("scaler",
            StandardScaler()
        ),
        ("classifier",
            RandomizedSearchCV(
                classifier,
                params,
                n_iter = 20,
                cv = 10,
                scoring = 'f1',
                return_train_score = True,
                random_state = 42,
                n_jobs = -1
            )
        )
    ]).fit(X_train, y_train)

def plot(x_axis_values, x_axis_values_order, y_axes_values, title, xlabel, ylabel, add_confidence_interval=True):
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)

    if x_axis_values_order == "asc":
        plt.xlim(min(x_axis_values)-0.05*max(x_axis_values), max(x_axis_values)+0.05*max(x_axis_values))
    elif x_axis_values_order == "dsec":
        plt.xlim(max(x_axis_values)+0.05*max(x_axis_values), min(x_axis_values)-0.05*max(x_axis_values))
    else:
        raise Exception("Invalid value for x_axis_values_order. Valid values are 'asc' and 'dsec'")

    for y_axis_values in y_axes_values:
        plt.plot(x_axis_values,
            y_axis_values["mean"],
            y_axis_values["marker"],
            alpha=0.8,
            color=y_axis_values["color"],
            label=y_axis_values["label"])

        if add_confidence_interval:
            plt.fill_between(list(x_axis_values),
                y_axis_values["mean"]-y_axis_values["std"],
                y_axis_values["mean"]+y_axis_values["std"],
                alpha=0.1,
                color=y_axis_values["color"])

    legend = plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", prop={"size": 14})

    plt.savefig(str(uuid.uuid1()), bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.show()

def get_model_complexity(model, parameter_complexity_order = "asc", model_name="Insert Model Name"):
    parameter_name = "param_" + list(model["classifier"].param_distributions.keys())[0]

    df = pd.DataFrame(model["classifier"].cv_results_, 
        index=model["classifier"].cv_results_[parameter_name])[["mean_test_score",
            "std_test_score",
            "mean_train_score",
            "std_train_score",
            parameter_name]].sort_index(ascending=(parameter_complexity_order == "asc"))

    mean_test_score = df["mean_test_score"]
    std_test_score = df["std_test_score"]
    mean_train_score = df["mean_train_score"]
    std_train_score = df["std_train_score"]
    parameter_values = [v[0] if isinstance(v, Sequence) else v for v in df[parameter_name]]

    return {"mean_test_score": mean_test_score,
        "std_test_score": std_test_score,
        "mean_train_score": mean_train_score,
        "std_train_score": std_train_score,
        "parameter_name": parameter_name.lstrip("param").lstrip("_"),
        "parameter_values": parameter_values,
        "parameter_complexity_order": parameter_complexity_order,
        "model_name": model_name
    }

def plot_model_complexity(model_complexity, marker):
    plot(model_complexity["parameter_values"],
        model_complexity["parameter_complexity_order"],
        [
            {
                "mean": model_complexity["mean_test_score"],
                "std": model_complexity["std_test_score"],
                "color": "r",
                "label": "Cross-Validation Score: " + model_complexity["model_name"],
                "marker": marker
            },
            {
                "mean": model_complexity["mean_train_score"],
                "std": model_complexity["std_train_score"],
                "color": "b",
                "label": "Training Score: " + model_complexity["model_name"],
                "marker": marker
            }
        ],
        "Model Complexity Curves",
        "Parameter: " + model_complexity["parameter_name"],
        "F1 Score")

def get_learning_curve(model, X_train, y_train, model_name="Insert Model Name"):
    mean_train_score = []
    std_train_score = []
    mean_test_score = []
    std_test_score = []

    mean_fit_time = []
    std_fit_time = []
    mean_score_time = []
    std_score_time = []

    n_training_examples = (np.linspace(0.05, 1.0, 20)*len(y_train)).astype("int")
    
    for i in n_training_examples:
        X_subset = X_train.sample(n=i, random_state=42)
        y_subset = y_train.sample(n=i, random_state=42)

        scores = cross_validate(Pipeline([
                ("preprocessor",
                    model["preprocessor"]
                ),
                ("scaler",
                    model["scaler"]
                ),
                ("classifier",
                    model["classifier"].best_estimator_
                )
            ]),
            X_subset,
            y_subset,
            cv=10,
            scoring="f1",
            n_jobs=-1,
            return_train_score=True)

        mean_train_score.append(np.mean(scores["train_score"]))
        std_train_score.append(np.std(scores["train_score"]))
        mean_test_score.append(np.mean(scores["test_score"]))
        std_test_score.append(np.std(scores["test_score"]))

        mean_fit_time.append(np.mean(scores["fit_time"]))
        std_fit_time.append(np.std(scores["fit_time"]))
        mean_score_time.append(np.mean(scores["score_time"]))
        std_score_time.append(np.std(scores["score_time"]))

    mean_train_score = np.array(mean_train_score)
    std_train_score = np.array(std_train_score)
    mean_test_score = np.array(mean_test_score)
    std_test_score = np.array(std_test_score)

    mean_fit_time = np.array(mean_fit_time)
    std_fit_time = np.array(std_fit_time)
    mean_score_time = np.array(mean_score_time)
    std_score_time = np.array(std_score_time)

    return {"mean_train_score": mean_train_score,
        "std_train_score": std_train_score,
        "mean_test_score": mean_test_score,
        "std_test_score": std_test_score,
        "mean_fit_time": mean_fit_time,
        "std_fit_time": std_fit_time,
        "mean_score_time": mean_score_time,
        "std_score_time": std_score_time,
        "n_training_examples": n_training_examples,
        "model_name": model_name
    }

def plot_learning_curve(learning_curve, marker):
    plot(learning_curve["n_training_examples"],
        "asc",
        [
            {
                "mean": learning_curve["mean_test_score"],
                "std": learning_curve["std_test_score"],
                "color": "r",
                "label": "Cross-Validation Score: " + learning_curve["model_name"],
                "marker": marker
            },
            {
                "mean": learning_curve["mean_train_score"],
                "std": learning_curve["std_train_score"],
                "color": "b",
                "label": "Training Score: " + learning_curve["model_name"],
                "marker": marker
            }
        ],
        "F1 Score Learning Curves",
        "No. Training Examples",
        "F1 Score")
    
    plot(learning_curve["n_training_examples"],
        "asc",
        [
            {
                "mean": learning_curve["mean_score_time"],
                "std": learning_curve["std_score_time"],
                "color": "r",
                "label": "Cross-Validation Time: " + learning_curve["model_name"],
                "marker": marker
            },
            {
                "mean": learning_curve["mean_fit_time"],
                "std": learning_curve["std_fit_time"],
                "color": "b",
                "label": "Training Time: " + learning_curve["model_name"],
                "marker": marker
            }
        ],
        "Time Learning Curves",
        "No. Training Examples",
        "Time (s)")

def get_model_evaluation_metrics(model, X_test, y_test, model_name="Insert Model Name"):
    y_predict = model.predict(X_test)

    return {
        "model_name": model_name,
        "auc": roc_auc_score(y_test, y_predict),
        "f1": f1_score(y_test, y_predict),
        "accuracy": accuracy_score(y_test, y_predict),
        "precision": precision_score(y_test, y_predict),
        "recall": recall_score(y_test, y_predict)
    }

def print_model_evaluation_metrics(model_evaluation_metrics):
    print(model_evaluation_metrics["model_name"] + " Evaluation Metrics On Out-Of-Sample Test Data")
    print("**********************************************************************")
    print("F1 Score:  "+"{:.2f}".format(model_evaluation_metrics["f1"]))
    print("Accuracy:  "+"{:.2f}".format(model_evaluation_metrics["accuracy"])+"     AUC:    "+"{:.2f}".format(model_evaluation_metrics["auc"]))
    print("Precision: "+"{:.2f}".format(model_evaluation_metrics["precision"])+"     Recall: "+"{:.2f}".format(model_evaluation_metrics["recall"]))
    print("**********************************************************************")

def plot_learning_curve_comparision(learning_curves):
    f1_scores = []
    times = []
    for learning_curve, color in zip(learning_curves, ['r', 'g', 'b',  'k', 'y', 'c']):
        f1_scores.append({
            "mean": learning_curve["mean_test_score"],
            "std": learning_curve["std_test_score"],
            "color": color,
            "label": "Cross-Validation Score: " + learning_curve["model_name"],
            "marker": "-"
        })
        times.append({
            "mean": learning_curve["mean_fit_time"],
            "std": learning_curve["std_fit_time"],
            "color": color,
            "label": "Training Time: " + learning_curve["model_name"],
            "marker": "-"
        })

    plot(learning_curve["n_training_examples"],
        "asc",
        f1_scores,
        "F1 Score Learning Curves Comparisions",
        "No. Training Examples",
        "F1 Score",
        False)

    plot(learning_curve["n_training_examples"],
        "asc",
        times,
        "Time Learning Curves Comparisions",
        "No. Training Examples",
        "Time (s)",
        False)

def print_model_evaluation_metrics_comparision(model_evaluation_metrices):
    print (pd.DataFrame(model_evaluation_metrices))
