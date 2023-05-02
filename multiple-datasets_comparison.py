from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from sklearn.tree import DecisionTreeClassifier

from fuzzytree import FuzzyDecisionTreeClassifier

import time
import pandas as pd


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

# Maybe add more metrics? https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
for name, data in data_sets:
    print("~"*3)
    print("Zbór danych:", name)
    print()

    X_train, X_test, y_train, y_test = train_test_split(
        data[0], data[1], test_size=0.2, random_state=42)

    t_start = time.time()
    clf_sk = DecisionTreeClassifier(
        max_depth=max_depth).fit(X_train, y_train)
    t_end = time.time()

    p_start = time.time()
    pred_sk = clf_sk.predict(X_test)
    p_end = time.time()

    print("Klasyczne drzewo decyzyjne:")
    print("Czas budowy:", (t_end - t_start) * 1000, "milisekund")

    print("Czas użycia:", (p_end - p_start) * 1000, "milisekund")

    print("Dokładność:", accuracy_score(y_test, pred_sk))

    print("Precyzja:", precision_score(
        y_test, pred_sk, average='macro', zero_division=0))

    print("Czułość:", recall_score(
        y_test, pred_sk, average='macro', zero_division=0))

    print("Macież błędów:\n", confusion_matrix(y_test, pred_sk))
    print()

    t_start = time.time()
    clf_fuzz = FuzzyDecisionTreeClassifier(
        max_depth=max_depth).fit(X_train, y_train)
    t_end = time.time()

    p_start = time.time()
    pred_fuzz = clf_fuzz.predict(X_test)
    p_end = time.time()

    print("Rozmyte drzewo decyzyjne:")
    print("Czas budowy:", (t_end - t_start) * 1000, "milisekund")

    print("Czas użycia:", (p_end - p_start) * 1000, "milisekund")

    print("Dokładnść:", accuracy_score(y_test, pred_fuzz))

    print("Precyzja:", precision_score(
        y_test, pred_fuzz, average='macro', zero_division=0))  # TODO: może jednak micro?

    print("Czułość:", recall_score(
        y_test, pred_fuzz, average='macro', zero_division=0))

    print("Macież błędów:\n", confusion_matrix(y_test, pred_fuzz))

    print("~"*3)
