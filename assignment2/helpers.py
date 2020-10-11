import mlrose

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import uuid
import time

from algorithms import *
from neural import *

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

plt.style.use("seaborn-whitegrid")

restarts = [10]
decays = [0.90, 0.92, 0.94, 0.96, 0.98]
mutation_probs = [0.01, 0.05, 0.1, 0.15, 0.2]
keep_pcts = [0.01, 0.05, 0.1, 0.15, 0.2]
hidden_units = [6]
n=3

def optimize(problem):
    hill_climb_best_fitnesses = []
    simulated_annealing_best_fitnesses = []
    genetic_alg_best_fitnesses = []
    mimic_best_fitnesses = []

    hill_climb_learning_curves = []
    simulated_annealing_learning_curves = []
    genetic_alg_learning_curves = []
    mimic_learning_curves = []

    hill_climb_execution_times = []
    simulated_annealing_execution_times = []
    genetic_alg_execution_times = []
    mimic_execution_times = []

    for i in range(n):
        _hill_climb_best_fitnesses = []
        _hill_climb_learning_curves = []
        _hill_climb_execution_times = []
        for restart in restarts:
            start = time.process_time()
            _, hill_climb_best_fitness, hill_climb_learning_curve = hill_climb(problem,
                restarts=restart,
                max_iters=10000,
                curve=True,
                random_state=i)
            end = time.process_time()

            _hill_climb_best_fitnesses.append(hill_climb_best_fitness)
            _hill_climb_learning_curves.append(hill_climb_learning_curve)
            _hill_climb_execution_times.append(end-start)
        
        hill_climb_best_fitnesses.append(_hill_climb_best_fitnesses)
        hill_climb_learning_curves.append(_hill_climb_learning_curves)
        hill_climb_execution_times.append(_hill_climb_execution_times)

        _simulated_annealing_best_fitnesses = []
        _simulated_annealing_learning_curves = []
        _simulated_annealing_execution_times = []
        for decay in decays:
            start = time.process_time()
            _, simulated_annealing_best_fitness, simulated_annealing_learning_curve = simulated_annealing(problem,
                schedule=mlrose.GeomDecay(init_temp=10.0, decay=decay, min_temp=0.001),
                max_attempts=100,
                max_iters=10000,
                curve=True,
                random_state=i)
            end = time.process_time()

            _simulated_annealing_best_fitnesses.append(simulated_annealing_best_fitness)
            _simulated_annealing_learning_curves.append(simulated_annealing_learning_curve)
            _simulated_annealing_execution_times.append(end-start)

        simulated_annealing_best_fitnesses.append(_simulated_annealing_best_fitnesses)
        simulated_annealing_learning_curves.append(_simulated_annealing_learning_curves)
        simulated_annealing_execution_times.append(_simulated_annealing_execution_times)

        _genetic_alg_best_fitnesses = []
        _genetic_alg_learning_curves = []
        _genetic_alg_execution_times = []
        for mutation_prob in mutation_probs:
            start = time.process_time()
            _, genetic_alg_best_fitness, genetic_alg_learning_curve = genetic_alg(problem,
                mutation_prob=mutation_prob,
                max_attempts=100,
                max_iters=10000,
                curve=True,
                random_state=i)
            end = time.process_time()

            _genetic_alg_best_fitnesses.append(genetic_alg_best_fitness)
            _genetic_alg_learning_curves.append(genetic_alg_learning_curve)
            _genetic_alg_execution_times.append(end-start)

        genetic_alg_best_fitnesses.append(_genetic_alg_best_fitnesses)
        genetic_alg_learning_curves.append(_genetic_alg_learning_curves)
        genetic_alg_execution_times.append(_genetic_alg_execution_times)

        _mimic_best_fitnesses = []
        _mimic_learning_curves = []
        _mimic_execution_times = []
        for keep_pct in keep_pcts:
            try:
                start = time.process_time()
                _, mimic_best_fitness, mimic_learning_curve = mimic(problem,
                    keep_pct=keep_pct,
                    max_attempts=100,
                    max_iters=10000,
                    curve=True,
                    random_state=i,
                    fast_mimic=True,
                    noise=0.1)
                end = time.process_time()
            except:
                start = time.process_time()
                _, mimic_best_fitness, mimic_learning_curve = mimic(problem,
                    keep_pct=keep_pct,
                    max_attempts=100,
                    max_iters=10000,
                    curve=True,
                    random_state=i,
                    noise=0.1)
                end = time.process_time()
        
            _mimic_best_fitnesses.append(mimic_best_fitness)
            _mimic_learning_curves.append(mimic_learning_curve)
            _mimic_execution_times.append(end-start)

        mimic_best_fitnesses.append(_mimic_best_fitnesses)
        mimic_learning_curves.append(_mimic_learning_curves)
        mimic_execution_times.append(_mimic_execution_times)
    
    return ([hill_climb_learning_curves, simulated_annealing_learning_curves, genetic_alg_learning_curves, mimic_learning_curves],
        [hill_climb_best_fitnesses, simulated_annealing_best_fitnesses, genetic_alg_best_fitnesses, mimic_best_fitnesses],
        [hill_climb_execution_times, simulated_annealing_execution_times, genetic_alg_execution_times, mimic_execution_times])

def aggregate_curves(arrays):
    arrays = np.array(arrays)
    means = []
    stds = []
    for j in range(arrays.shape[1]):
        max_len = max([len(a) for a in arrays[:,j]])
        aa = np.array([np.pad(a, (0, max_len - len(a)), 'constant', constant_values=np.nan) for a in arrays[:,j]])

        mean, std = np.nanmean(aa, axis=0), np.nanstd(aa, axis=0)
        means.append(mean)
        stds.append(std)

    return means, stds

def plot_curves(data, xlabel, ylabel, add_confidence_interval=True):
    gridx_len, gridy_len = len(data), len(data[0])

    fig, axs = plt.subplots(gridx_len, gridy_len, sharey=True)
    fig.text(0.5, 0.08, xlabel, ha='center', va='center', fontsize=12)
    fig.text(0.06, 0.5, ylabel, ha='center', va='center', rotation='vertical', fontsize=12)
    plt.subplots_adjust(top=0.84, hspace=0.42, wspace=0.6)

    legends = []
    for xi in range(gridx_len):
        for yi in range(gridy_len):
            axs[xi, yi].set_title(data[xi][yi]["title"])

            max_len = max([len(a) for a in data[xi][yi]["mean"]])
            data[xi][yi]["mean"] = np.array([np.pad(a, (0, max_len - len(a)), 'constant', constant_values=np.nan) for a in data[xi][yi]["mean"]])
            data[xi][yi]["std"] = np.array([np.pad(a, (0, max_len - len(a)), 'constant', constant_values=np.nan) for a in data[xi][yi]["std"]])

            for mean, std, color, label in zip(data[xi][yi]["mean"],
                data[xi][yi]["std"],
                ["r", "g", "c", "y", "b"][:len(data[xi][yi]["mean"])],
                data[xi][yi]["label"]):
                axs[xi, yi].plot(range(max_len),
                                 mean,
                                 "-",
                                 alpha=0.8,
                                 color=color,
                                 label=label)

                if add_confidence_interval:
                    axs[xi, yi].fill_between(list(range(max_len)),
                        mean-std,
                        mean+std,
                        alpha=0.2,
                        color=color)

            legends.append(axs[xi, yi].legend(bbox_to_anchor=(1.0,1.0), loc="upper left", prop={"size": 8}, title=data[xi][yi]["parameter"]))

    plt.savefig(str(uuid.uuid1()), bbox_extra_artists=legends, bbox_inches='tight', pad_inches=0.3, dpi=600)
    plt.show()

def plot_learning_curves(hill_climb_curves, simulated_annealing_curves, genetic_alg_curves, mimic_curves, is_mimic=True):
    hill_climb_curves_means, hill_climb_curves_stds = aggregate_curves(hill_climb_curves)
    simulated_annealing_curves_means, simulated_annealing_curves_stds = aggregate_curves(simulated_annealing_curves)
    genetic_alg_curves_means, genetic_alg_curves_stds = aggregate_curves(genetic_alg_curves)
    mimic_curves_means, mimic_curves_stds = aggregate_curves(mimic_curves)

    plot_curves([
        [
            {
                "mean": hill_climb_curves_means,
                "std": hill_climb_curves_stds,
                "title": "RHC",
                "parameter": "Restarts",
                "label": restarts
            },
            {
                "mean": simulated_annealing_curves_means,
                "std": simulated_annealing_curves_stds,
                "title": "SA",
                "parameter": "Geom Decay",
                "label": decays
            }
        ],
        [
            {
                "mean": genetic_alg_curves_means,
                "std": genetic_alg_curves_stds,
                "title": "GA",
                "parameter": "Mutation Prob",
                "label": mutation_probs
            },
            {
                "mean": mimic_curves_means,
                "std": mimic_curves_stds,
                "title": "MIMIC" if is_mimic else "Back Proposition",
                "parameter": "Keep Pct" if is_mimic else "Hidden Units",
                "label": keep_pcts if is_mimic else hidden_units
            }
        ]
    ],
    "Iteration",
    "Fitness Function Value")

def aggregate_bars(arrays):
    arrays = np.array(arrays)
    means = []
    stds = []
    for j in range(arrays.shape[1]):
        mean, std = np.nanmean(arrays[:,j], axis=0), np.nanstd(arrays[:,j], axis=0)
        means.append(mean)
        stds.append(std)

    return means, stds

def plot_bars(data, xlabel, ylabel, add_confidence_interval=True):
    fig, ax = plt.subplots()
    fig.text(0.5, 0.02, xlabel, ha='center', va='center', fontsize=12)
    fig.text(0.06, 0.5, ylabel, ha='center', va='center', rotation='vertical', fontsize=12)

    w = 1.0

    xticks = []
    xticklabels = []
    for di in range(len(data)):
        mean = data[di]["mean"]
        std = data[di]["std"]

        if di == 0:
            x = 1.0
        else:
            x += (len(data[di-1]["mean"])+len(data[di]["mean"]))/2.0 + w/2
        
        c = ["r", "g", "c", "y", "b"][di]

        for mi in range(len(mean)):
            xtick = x - (len(mean)-2*mi+1)*w/2 
            ax.bar(xtick, mean[mi], w,
                yerr=std[mi],
                color=c,
                edgecolor="k",
                label=data[di]["title"])

            if mean[mi] < 1:
                m = round(mean[mi], 3)
            elif mean[mi] < 10:
                m = round(mean[mi], 2)
            elif mean[mi] < 100:
                m = round(mean[mi], 1)
            else:
                m = round(mean[mi], 0)

            ax.annotate(m,
                xy=(xtick, mean[mi] + std[mi]),
                ha="center",
                va="bottom",
                rotation=60)

            data[di]["title"] = ""
            xticks.append(xtick)
            if int(len(mean)/2) == mi:
                xticklabels.append(str(data[di]["label"][mi]) + "\n" + data[di]["parameter"])
            else:
                xticklabels.append(str(data[di]["label"][mi]))
    
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=8)
    
    legend = ax.legend(bbox_to_anchor=(1.0,1.0), loc="upper left", prop={"size": 8}, title="Algorithm")

    plt.savefig(str(uuid.uuid1()), bbox_extra_artists=(legend,), bbox_inches='tight', pad_inches=0.3, dpi=600)
    plt.show()

def plot_best_fitnesses(hill_climb_best_fitnesses, simulated_annealing_best_fitnesses, genetic_alg_best_fitnesses, mimic_best_fitnesses, is_mimic=True):
    hill_climb_best_fitnesses_means, hill_climb_best_fitnesses_stds = aggregate_bars(hill_climb_best_fitnesses)
    simulated_annealing_means, simulated_annealing_best_fitnesses_stds = aggregate_bars(simulated_annealing_best_fitnesses)
    genetic_alg_best_fitnesses_means, genetic_alg_best_fitnesses_stds = aggregate_bars(genetic_alg_best_fitnesses)
    mimic_best_fitnesses_means, mimic_best_fitnesses_stds = aggregate_bars(mimic_best_fitnesses)

    plot_bars([
		{
			"mean": hill_climb_best_fitnesses_means,
			"std": hill_climb_best_fitnesses_stds,
			"title": "RHC",
			"parameter": "Restarts",
			"label": restarts if is_mimic else [0]
		},
		{
			"mean": simulated_annealing_means,
			"std": simulated_annealing_best_fitnesses_stds,
			"title": "SA",
			"parameter": "Geom Decay",
			"label": decays
		},
		{
			"mean": genetic_alg_best_fitnesses_means,
			"std": genetic_alg_best_fitnesses_stds,
			"title": "GA",
			"parameter": "Mutation Prob",
			"label": mutation_probs
		},
		{
			"mean": mimic_best_fitnesses_means,
			"std": mimic_best_fitnesses_stds,
			"title": "MIMIC" if is_mimic else "BP",
			"parameter": "Keep Pct" if is_mimic else "Hidden Units",
			"label": keep_pcts if is_mimic else hidden_units
		}
    ],
    "Algorithm Parameter",
    "Fitness Function Value")

def plot_execution_times(hill_climb_execution_times, simulated_annealing_execution_times, genetic_alg_execution_times, mimic_execution_times, is_mimic=True):
    hill_climb_execution_times_means, hill_climb_execution_times_stds = aggregate_bars(hill_climb_execution_times)
    simulated_annealing_means, simulated_annealing_execution_times_stds = aggregate_bars(simulated_annealing_execution_times)
    genetic_alg_execution_times_means, genetic_alg_execution_times_stds = aggregate_bars(genetic_alg_execution_times)
    mimic_execution_times_means, mimic_execution_times_stds = aggregate_bars(mimic_execution_times)

    plot_bars([
		{
			"mean": hill_climb_execution_times_means,
			"std": hill_climb_execution_times_stds,
			"title": "RHC",
			"parameter": "Restarts",
			"label": restarts if is_mimic else [0]
		},
		{
			"mean": simulated_annealing_means,
			"std": simulated_annealing_execution_times_stds,
			"title": "SA",
			"parameter": "Geom Decay",
			"label": decays
		},
		{
			"mean": genetic_alg_execution_times_means,
			"std": genetic_alg_execution_times_stds,
			"title": "GA",
			"parameter": "Mutation Prob",
			"label": mutation_probs
		},
		{
			"mean": mimic_execution_times_means,
			"std": mimic_execution_times_stds,
			"title": "MIMIC" if is_mimic else "BP",
			"parameter": "Keep Pct" if is_mimic else "Hidden Units",
			"label": keep_pcts if is_mimic else hidden_units
		}
    ],
    "Algorithm Parameter",
    "Time (Sec)")

def plot_hist(df, group_by, ohe={}):
    group_by_col = df[group_by]

    for old, new in ohe.items():
        loc = list(df.columns).index(old)
        values = LabelEncoder().fit_transform(df[old])
        df = df.drop([old], axis='columns')
        df.insert(loc=loc, column=new, value=values)

    grouped_df = df.groupby(group_by_col)

    _, axs = plt.subplots(int(len(df.columns)/5 + 0.5), 5)
    for i in range(len(df.columns)):
        value_counts = len(grouped_df[df.columns[i]].value_counts())
        grouped_df[df.columns[i]].plot(ax=axs[i//5, i%5],
                                kind='hist',
                                bins=value_counts if value_counts < 100 else 100,
                                figsize=(15, 9),
                                alpha=0.7,
                                title=df.columns[i] + " (" + str(df['class'].isnull().sum()) + " nulls)",
                                legend=True)

    plt.savefig(str(uuid.uuid1()), bbox_inches='tight', pad_inches=0.3, dpi=600)
    plt.show()

def fit(X, y):
    hill_climb_best_fitnesses = []
    simulated_annealing_best_fitnesses = []
    genetic_alg_best_fitnesses = []
    backprop_best_fitnesses = []

    hill_climb_learning_curves = []
    simulated_annealing_learning_curves = []
    genetic_alg_learning_curves = []
    backprop_learning_curves = []

    hill_climb_execution_times = []
    simulated_annealing_execution_times = []
    genetic_alg_execution_times = []
    backprop_execution_times = []

    for i in range(n):
        _hill_climb_best_fitnesses = []
        _hill_climb_learning_curves = []
        _hill_climb_execution_times = []
        for restart in [0]:
            model = NeuralNetwork(
                algorithm='random_hill_climb',
                hidden_nodes = [6],
                activation='sigmoid',
                clip_max=5,
                max_iters=10000,
                early_stopping=True,
                restarts=restart,
                learning_rate=0.5,
                curve=True,
                random_state=i)

            start = time.process_time()
            model = model.fit(X, y)
            end = time.process_time()

            _hill_climb_best_fitnesses.append(model.loss)
            _hill_climb_learning_curves.append(model.fitness_curve)
            _hill_climb_execution_times.append(end-start)
        
        hill_climb_best_fitnesses.append(_hill_climb_best_fitnesses)
        hill_climb_learning_curves.append(_hill_climb_learning_curves)
        hill_climb_execution_times.append(_hill_climb_execution_times)

        _simulated_annealing_best_fitnesses = []
        _simulated_annealing_learning_curves = []
        _simulated_annealing_execution_times = []
        for decay in decays:
            model = NeuralNetwork(
                algorithm='simulated_annealing',
                hidden_nodes = [6],
                activation='sigmoid',
                clip_max=5,
                max_iters=10000,
                early_stopping=True,
                max_attempts=100,
                schedule=mlrose.GeomDecay(init_temp=10.0, decay=decay, min_temp=0.001),
                learning_rate=0.5,
                curve=True,
                random_state=i)

            start = time.process_time()
            model = model.fit(X, y)
            end = time.process_time()

            _simulated_annealing_best_fitnesses.append(model.loss)
            _simulated_annealing_learning_curves.append(model.fitness_curve)
            _simulated_annealing_execution_times.append(end-start)

        simulated_annealing_best_fitnesses.append(_simulated_annealing_best_fitnesses)
        simulated_annealing_learning_curves.append(_simulated_annealing_learning_curves)
        simulated_annealing_execution_times.append(_simulated_annealing_execution_times)

        _genetic_alg_best_fitnesses = []
        _genetic_alg_learning_curves = []
        _genetic_alg_execution_times = []
        for mutation_prob in mutation_probs:
            model = NeuralNetwork(
                algorithm='genetic_alg',
                hidden_nodes = [6],
                activation='sigmoid',
                clip_max=5,
                max_iters=10000,
                early_stopping=True,
                max_attempts=100,
                mutation_prob=mutation_prob,
                learning_rate=0.5,
                curve=True,
                random_state=i)

            start = time.process_time()
            model = model.fit(X, y)
            end = time.process_time()

            _genetic_alg_best_fitnesses.append(model.loss)
            _genetic_alg_learning_curves.append(model.fitness_curve)
            _genetic_alg_execution_times.append(end-start)

        genetic_alg_best_fitnesses.append(_genetic_alg_best_fitnesses)
        genetic_alg_learning_curves.append(_genetic_alg_learning_curves)
        genetic_alg_execution_times.append(_genetic_alg_execution_times)

        model = MLPClassifier(activation='logistic',
                    learning_rate_init=0.005,
                    max_iter=500,
                    hidden_layer_sizes=(6,),
                    random_state=i)

        start = time.process_time()
        model = model.fit(X, y)
        end = time.process_time()

        backprop_best_fitnesses.append([model.loss_])
        backprop_learning_curves.append([model.loss_curve_])
        backprop_execution_times.append([end-start])
    
    return ([hill_climb_learning_curves, simulated_annealing_learning_curves, genetic_alg_learning_curves, backprop_learning_curves],
        [hill_climb_best_fitnesses, simulated_annealing_best_fitnesses, genetic_alg_best_fitnesses, backprop_best_fitnesses],
        [hill_climb_execution_times, simulated_annealing_execution_times, genetic_alg_execution_times, backprop_execution_times])

def plot_evaluation_matrix(X_train, y_train, X_test, y_test, restarts, decay, mutation_prob):
    metrices = []

    model = NeuralNetwork(
        algorithm='random_hill_climb',
        hidden_nodes = [6],
        activation='sigmoid',
        clip_max=5,
        max_iters=1000,
        early_stopping=True,
        restarts=restarts,
        learning_rate=0.5,
        curve=True,
        random_state=2)

    model = model.fit(X_train, y_train)

    y_predict = model.predict(X_test)
    metrices.append({
        "auc": roc_auc_score(y_test, y_predict),
        "f1": f1_score(y_test, y_predict),
        "accuracy": accuracy_score(y_test, y_predict),
        "precision": precision_score(y_test, y_predict),
        "recall": recall_score(y_test, y_predict)
    })

    model = NeuralNetwork(
        algorithm='simulated_annealing',
        hidden_nodes = [6],
        activation='sigmoid',
        clip_max=5,
        max_iters=10000,
        early_stopping=True,
        max_attempts=100,
        schedule=mlrose.GeomDecay(init_temp=10.0, decay=decay, min_temp=0.001),
        learning_rate=0.5,
        curve=True,
        random_state=2)

    model = model.fit(X_train, y_train)

    y_predict = model.predict(X_test)
    metrices.append({
        "auc": roc_auc_score(y_test, y_predict),
        "f1": f1_score(y_test, y_predict),
        "accuracy": accuracy_score(y_test, y_predict),
        "precision": precision_score(y_test, y_predict),
        "recall": recall_score(y_test, y_predict)
    })

    model = NeuralNetwork(
        algorithm='genetic_alg',
        hidden_nodes = [6],
        activation='sigmoid',
        clip_max=5,
        max_iters=10000,
        early_stopping=True,
        max_attempts=100,
        mutation_prob=mutation_prob,
        learning_rate=0.5,
        curve=True,
        random_state=2)

    model = model.fit(X_train, y_train)

    y_predict = model.predict(X_test)
    metrices.append({
        "auc": roc_auc_score(y_test, y_predict),
        "f1": f1_score(y_test, y_predict),
        "accuracy": accuracy_score(y_test, y_predict),
        "precision": precision_score(y_test, y_predict),
        "recall": recall_score(y_test, y_predict)
    })

    model = MLPClassifier(activation='logistic',
                learning_rate_init=0.005,
                max_iter=500,
                hidden_layer_sizes=(6,),
                random_state=2)

    model = model.fit(X_train, y_train)

    y_predict = model.predict(X_test)
    metrices.append({
        "auc": roc_auc_score(y_test, y_predict),
        "f1": f1_score(y_test, y_predict),
        "accuracy": accuracy_score(y_test, y_predict),
        "precision": precision_score(y_test, y_predict),
        "recall": recall_score(y_test, y_predict)
    })

    pd.DataFrame(metrices, index=["RCH", "SA", "GA", "BP"]).style.background_gradient(cmap ='YlGn').set_properties(**{'font-size': '20px'})
