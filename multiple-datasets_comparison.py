from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles
from sklearn.datasets import load_iris
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier

from fuzzytree import FuzzyDecisionTreeClassifier

import time

iris = load_iris()

X_iris = iris.data[:, ]
y_iris = iris.target

data_sets = [
    ('blobs',
     make_blobs(
         n_samples=300,
         n_features=2,
         centers=[[0, 5],
                  [10, 20],
                  [20, 5]],
         cluster_std=[10, 5, 10],
         random_state=42)),

    ('circles',
     make_circles(
         n_samples=300,
         noise=0.3,
         factor=0.4,
         random_state=42)),

    ('iris',
     (X_iris,
      y_iris)),

    ('moons',
     make_moons(
         n_samples=300,
         noise=0.5,
         random_state=42))
]

# Maybe add more metrics? https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
for name, data in data_sets:
    print("~"*3)
    print("Zbór danych:", name)
    print()

    X_train, X_test, y_train, y_test = train_test_split(
        data[0], data[1], test_size=0.2, random_state=42)

    t_start = time.time()
    clf_fuzz = FuzzyDecisionTreeClassifier().fit(X_train, y_train)
    t_end = time.time()

    p_start = time.time()
    pred_fuzz = clf_fuzz.predict(X_test)
    p_end = time.time()

    print("Rozmyte drzewo decyzyjne:")
    print("Czas budowy:", (t_end - t_start) * 1000, "milisekund")
    print("Czas użycia:", (p_end - p_start) * 1000, "milisekund")
    print("Dokładnść:", accuracy_score(y_test, pred_fuzz))
    print("Macież błędów:\n", confusion_matrix(y_test, pred_fuzz))
    print()

    t_start = time.time()
    clf_sk = DecisionTreeClassifier().fit(X_train, y_train)
    t_end = time.time()

    p_start = time.time()
    pred_sk = clf_sk.predict(X_test)
    p_end = time.time()

    print("Klasyczne drzewo decyzyjne:")
    print("Czas budowy:", (t_end - t_start) * 1000, "milisekund")
    print("Czas użycia:", (p_end - p_start) * 1000, "milisekund")
    print("Dokładność:", accuracy_score(y_test, pred_sk))
    print("Macież błędów:\n", confusion_matrix(y_test, pred_sk))

    print("~"*3)
