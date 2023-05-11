from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

from fuzzytree import FuzzyDecisionTreeClassifier

import time
import pandas as pd
import matplotlib.pyplot as plt


def read_dataset(dataset_name, header='infer', index_col=None, delimiter=',', labels_encoding=[], get_dummies=False):
    df = pd.read_csv('./datasets/' + dataset_name + '.csv',
                     header=header, index_col=index_col, delimiter=delimiter)

    if labels_encoding:
        for label in labels_encoding:
            encoder = LabelEncoder()
            df[label] = encoder.fit_transform(df[label])

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1:]

    if get_dummies:
        X = pd.get_dummies(X, dtype=int)

    y = y.to_numpy().ravel()

    return (dataset_name, (X, y))


data_sets = [
    read_dataset('animals', 0, 0),
    read_dataset('fetal_health'),
    read_dataset('glass'),
    read_dataset('heart_attack'),
    read_dataset('mobile_price'),
    read_dataset('gender', labels_encoding=['gender']),
    read_dataset('oil_spill'),
    read_dataset('diabetes', delimiter=';', labels_encoding=['gender']),
    read_dataset('drugs', get_dummies=True, labels_encoding=['Drug']),
    read_dataset('wine', labels_encoding=['quality']),
]

max_depth = None


def make_bigger_bold(a, b):
    if a > b:
        a = '**' + str(a) + '**'
    elif b > a:
        b = '**' + str(b) + '**'

    return a, b


def calc_metrics(y, pred):
    return (
        accuracy_score(y, pred),
        precision_score(y, pred, average='macro', zero_division=0),
        precision_score(y, pred, average='micro', zero_division=0),
        recall_score(y, pred, average='macro', zero_division=0),
        recall_score(y, pred, average='micro', zero_division=0),
        f1_score(y, pred, average='macro', zero_division=0),
        f1_score(y, pred, average='micro', zero_division=0)
    )


for name, data in data_sets:
    print("## Zbiór danych:", name)
    print("### Statystyki zbioru")

    X = data[0]
    Y = data[1]

    desc = X.describe().T
    print(desc.to_markdown())
    print("### Metryki klasyfikatorów")

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42)

    t_start = time.time()
    clf_sk = DecisionTreeClassifier(
        max_depth=max_depth).fit(X_train, y_train)
    t_end = time.time()
    t_time_ms_sk = (t_end - t_start) * 1000

    p_start = time.time()
    pred_sk = clf_sk.predict(X_test)
    p_end = time.time()
    p_time_ms_sk = (p_end - p_start) * 1000

    (acc_sk,
     prec_macro_sk,
     prec_micro_sk,
     rec_macro_sk,
     rec_micro_sk,
     f1_macro_sk,
     f1_micro_sk) = calc_metrics(y_test, pred_sk)

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

    (acc_fuzz,
     prec_macro_fuzz,
     prec_micro_fuzz,
     rec_macro_fuzz,
     rec_micro_fuzz,
     f1_macro_fuzz,
     f1_micro_fuzz) = calc_metrics(y_test, pred_fuzz)

    cm_fuzz = confusion_matrix(y_test, pred_fuzz)
    disp_fuzz = ConfusionMatrixDisplay(confusion_matrix=cm_fuzz)
    disp_fuzz.plot()
    cm_fuzz_plot_path = './images/' + name + '_fuzz.png'
    plt.savefig(cm_fuzz_plot_path, bbox_inches='tight')

    cm_fuzz_str = '<br>'.join([', '.join([str(cell) for cell in row])
                               for row in cm_fuzz])

    acc_sk, acc_fuzz = make_bigger_bold(
        acc_sk, acc_fuzz)

    prec_macro_sk, prec_macro_fuzz = make_bigger_bold(
        prec_macro_sk, prec_macro_fuzz)

    prec_micro_sk, prec_micro_fuzz = make_bigger_bold(
        prec_micro_sk, prec_micro_fuzz)

    rec_macro_sk, rec_macro_fuzz = make_bigger_bold(
        rec_macro_sk, rec_macro_fuzz)

    rec_micro_sk, rec_micro_fuzz = make_bigger_bold(
        rec_micro_sk, rec_micro_fuzz)

    f1_macro_sk, f1_macro_fuzz = make_bigger_bold(
        f1_macro_sk, f1_macro_fuzz)

    f1_micro_sk, f1_micro_fuzz = make_bigger_bold(
        f1_micro_sk, f1_micro_fuzz)

    print("| Metryka | Klasyczne drzewo decyzyjne | Rozmyte drzewo decyzyjne ")
    print("| --- | --- | --- ")
    print("| Czas budowy [ms] | ", t_time_ms_sk, " | ", t_time_ms_fuzz)
    print("| Czas użycia [ms] | ", p_time_ms_sk, " | ", p_time_ms_fuzz)
    print("| Dokładność | ", acc_sk, " | ", acc_fuzz)
    print("| Precyzja makro |", prec_macro_sk, " | ", prec_macro_fuzz)
    print("| Precyzja mikro |", prec_micro_sk, " | ", prec_micro_fuzz)
    print("| Czułość makro |", rec_macro_sk, " | ", rec_macro_fuzz)
    print("| Czułość mikro |", rec_micro_sk, " | ", rec_micro_fuzz)
    print("| F1 makro |", f1_macro_sk, " | ", f1_macro_fuzz)
    print("| F1 mikro |", f1_micro_sk, " | ", f1_micro_fuzz)
    print(
        "| Macierz błędów |", "![](", cm_sk_plot_path, ")", " | ",  "![](", cm_fuzz_plot_path, ")")
    print("\n---\n")
