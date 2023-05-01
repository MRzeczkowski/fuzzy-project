# Porównanie rozmytego i klasycznego drzewa decyzyjnego

## Wstęp

Celem tego projektu było porównanie wybranych parametrów klasyfikacji za pomocą rozmytego i klasycznego drzewa decyzyjnego. W tym celu wykorzystano dziesięć zbiorów danych pobranych ze strony [Kaggle](https://www.kaggle.com/), za pomocą bibliotek [fuzzy-tree](https://balins.github.io/fuzzytree/index.html) i [Sklearn](https://scikit-learn.org/stable/) zbudowano odpowiednio rozmyte i klasyczne drzewo decyzyjne oraz za pomocą skryptu w języku Python wygenerowano raport z wybranymi statystykami.


## Opis zbiorów danych


## Algorytm budowania rozmytego drzewa decyzyjnego


## Raport
Poniższy Raport zawiera dla każdego zbioru danych informacje o tym jak długo trwało budowanie drzew, ich użycie, dokładność klasyfikacji na zbiorze testującym oraz macież błędów która może zostać użyta do dalszej analizy.

~~~
Dataset name: blobs

Fuzzy decision tree statistics:
Training time:  2486.3979816436768 miliseconds
Prediction time:  3.192901611328125 miliseconds
Accuracy:  0.8166666666666667
Confusion matrix: 
 [[14  2  1]
 [ 1 23  2]
 [ 3  2 12]]

Decision tree statistics:
Training time:  2.5262832641601562 miliseconds
Prediction time:  0.35881996154785156 miliseconds
Accuracy:  0.7833333333333333
Confusion matrix: 
 [[13  2  2]
 [ 1 23  2]
 [ 6  0 11]]
~~~
~~~
Dataset name: circles

Fuzzy decision tree statistics:
Training time:  2712.785243988037 miliseconds
Prediction time:  3.02886962890625 miliseconds
Accuracy:  0.8833333333333333
Confusion matrix: 
 [[24  2]
 [ 5 29]]

Decision tree statistics:
Training time:  0.4420280456542969 miliseconds
Prediction time:  0.06413459777832031 miliseconds
Accuracy:  0.8
Confusion matrix: 
 [[22  4]
 [ 8 26]]
~~~
~~~
Dataset name: iris

Fuzzy decision tree statistics:
Training time:  479.9990653991699 miliseconds
Prediction time:  1.1649131774902344 miliseconds
Accuracy:  1.0
Confusion matrix: 
 [[10  0  0]
 [ 0  9  0]
 [ 0  0 11]]

Decision tree statistics:
Training time:  0.2989768981933594 miliseconds
Prediction time:  0.06413459777832031 miliseconds
Accuracy:  1.0
Confusion matrix: 
 [[10  0  0]
 [ 0  9  0]
 [ 0  0 11]]
~~~
~~~
Dataset name: moons

Fuzzy decision tree statistics:
Training time:  2365.899085998535 miliseconds
Prediction time:  2.9478073120117188 miliseconds
Accuracy:  0.8333333333333334
Confusion matrix: 
 [[20  6]
 [ 4 30]]

Decision tree statistics:
Training time:  0.4069805145263672 miliseconds
Prediction time:  0.05984306335449219 miliseconds
Accuracy:  0.7166666666666667
Confusion matrix: 
 [[15 11]
 [ 6 28]]
~~~

## Podsumowanie
