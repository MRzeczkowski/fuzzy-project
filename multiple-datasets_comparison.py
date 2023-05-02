from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier

from fuzzytree import FuzzyDecisionTreeClassifier

import time
import pandas as pd
import matplotlib.pyplot as plt


def read_dataset(dataset_name, header='infer', index_col=None):
    df = pd.read_csv('./datasets/' + dataset_name + '.csv',
                     header=header, index_col=index_col)
    X = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1:].to_numpy().ravel()

    return (dataset_name, (X, y))


data_sets = [
    read_dataset('animals', 0, 0),
    read_dataset('fetal_health'),
    read_dataset('glass'),
    read_dataset('heart_attack'),
    read_dataset('mobile_price_train'),
]

max_depth = 64


def make_bigger_bold(a, b):
    if a > b:
        a = '**' + str(a) + '**'
    elif b > a:
        b = '**' + str(b) + '**'

    return a, b


for name, data in data_sets:
    print("## Zbiór danych:", name)
    print()

    X_train, X_test, y_train, y_test = train_test_split(
        data[0], data[1], test_size=0.2, random_state=42)

    t_start = time.time()
    clf_sk = DecisionTreeClassifier(
        max_depth=max_depth).fit(X_train, y_train)
    t_end = time.time()
    t_time_ms_sk = (t_end - t_start) * 1000

    p_start = time.time()
    pred_sk = clf_sk.predict(X_test)
    p_end = time.time()
    p_time_ms_sk = (p_end - p_start) * 1000

    acc_sk = accuracy_score(y_test, pred_sk)
    prec_sk = precision_score(
        y_test, pred_sk, average='macro', zero_division=0)  # TODO: micro?
    rec_sk = recall_score(
        y_test, pred_sk, average='macro', zero_division=0)

    cm_sk = confusion_matrix(y_test, pred_sk)
    disp_sk = ConfusionMatrixDisplay(confusion_matrix=cm_sk)
    disp_sk.plot()
    cm_sk_plot_path = './images/' + name + '_sk.png'
    plt.savefig(cm_sk_plot_path, bbox_inches='tight')

    cm_sk_str = '<br>'.join([', '.join([str(cell) for cell in row])
                             for row in cm_sk])

    t_start = time.time()
    clf_fuzz = FuzzyDecisionTreeClassifier(
        max_depth=max_depth).fit(X_train, y_train)
    t_end = time.time()
    t_time_ms_fuzz = (t_end - t_start) * 1000

    p_start = time.time()
    pred_fuzz = clf_fuzz.predict(X_test)
    p_end = time.time()
    p_time_ms_fuzz = (p_end - p_start) * 1000

    acc_fuzz = accuracy_score(y_test, pred_fuzz)
    prec_fuzz = recall_score(
        y_test, pred_fuzz, average='macro', zero_division=0)
    rec_fuzz = recall_score(
        y_test, pred_fuzz, average='macro', zero_division=0)

    cm_fuzz = confusion_matrix(y_test, pred_fuzz)
    disp_fuzz = ConfusionMatrixDisplay(confusion_matrix=cm_fuzz)
    disp_fuzz.plot()
    cm_fuzz_plot_path = './images/' + name + '_fuzz.png'
    plt.savefig(cm_fuzz_plot_path, bbox_inches='tight')

    cm_fuzz_str = '<br>'.join([', '.join([str(cell) for cell in row])
                               for row in cm_fuzz])

    acc_sk, acc_fuzz = make_bigger_bold(
        acc_sk, acc_fuzz)

    prec_sk, prec_fuzz = make_bigger_bold(
        prec_sk, prec_fuzz)

    rec_sk, rec_fuzz = make_bigger_bold(
        rec_sk, rec_fuzz)

    print("| Metryka | Klasyczne drzewo decyzyjne | Rozmyte drzewo decyzyjne ")
    print("| ------- | -------------------------- | ------------------------ ")
    print("| Czas budowy [ms] | ", t_time_ms_sk, " | ", t_time_ms_fuzz)
    print("| Czas użycia [ms] | ", p_time_ms_sk, " | ", p_time_ms_fuzz)
    print("| Dokładność | ", acc_sk, " | ", acc_fuzz)
    print("| Precyzja |", prec_sk, " | ", prec_fuzz)
    print("| Czułość |", rec_sk, " | ", rec_fuzz)
    print(
        "| Macierz błędów |", "![](", cm_sk_plot_path, ")", " | ",  "![](", cm_fuzz_plot_path, ")")
    print("\n---\n")
