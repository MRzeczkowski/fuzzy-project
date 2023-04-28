from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles
from sklearn.datasets import load_iris
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier

from fuzzytree import FuzzyDecisionTreeClassifier

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
    print("Dataset name: " + name)

    X_train, X_test, y_train, y_test = train_test_split(
        data[0], data[1], test_size=0.2, random_state=42)

    clf_fuzz = FuzzyDecisionTreeClassifier().fit(X_train, y_train)
    clf_sk = DecisionTreeClassifier().fit(X_train, y_train)

    pred_fuzz = clf_fuzz.predict(X_test)

    print("Fuzzy decision tree statistics:")
    print("Confusion matrix: \n" + str(confusion_matrix(y_test, pred_fuzz)))
    print("Accuracy: " + str(accuracy_score(y_test, pred_fuzz)))
    print()

    pred_sk = clf_sk.predict(X_test)

    print("Decision tree statistics:")
    print("Confusion matrix: \n" + str(confusion_matrix(y_test, pred_sk)))
    print("Accuracy: " + str(accuracy_score(y_test, pred_sk)))

    print("~"*100)
