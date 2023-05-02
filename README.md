# Porównanie rozmytego i klasycznego drzewa decyzyjnego

## Wstęp

Celem tego projektu było porównanie wybranych parametrów klasyfikacji za pomocą rozmytego i klasycznego drzewa decyzyjnego. W tym celu wykorzystano dziesięć zbiorów danych pobranych ze strony [Kaggle](https://www.kaggle.com/), za pomocą bibliotek [fuzzy-tree](https://balins.github.io/fuzzytree/index.html) i [Sklearn](https://scikit-learn.org/stable/) zbudowano odpowiednio rozmyte i klasyczne drzewo decyzyjne oraz za pomocą skryptu w języku Python wygenerowano raport z wybranymi statystykami.


## Opis zbiorów danych


## Algorytm budowania rozmytego drzewa decyzyjnego


## Raport
Poniższy Raport zawiera dla każdego zbioru danych informacje o tym jak długo trwało budowanie drzew, ich użycie, dokładność klasyfikacji na zbiorze testującym oraz macież błędów która może zostać użyta do dalszej analizy.

~~~
Zbór danych: blobs

Rozmyte drzewo decyzyjne:
Czas budowy: 2495.400905609131 milisekund
Czas użycia: 3.3380985260009766 milisekund
Dokładnść: 0.8166666666666667
Macież błędów:
 [[14  2  1]
 [ 1 23  2]
 [ 3  2 12]]

Klasyczne drzewo decyzyjne:
Czas budowy: 2.443075180053711 milisekund
Czas użycia: 0.31304359436035156 milisekund
Dokładność: 0.7333333333333333
Macież błędów:
 [[11  3  3]
 [ 2 22  2]
 [ 6  0 11]]
~~~
~~~
Zbór danych: circles

Rozmyte drzewo decyzyjne:
Czas budowy: 2695.902109146118 milisekund
Czas użycia: 3.0939579010009766 milisekund
Dokładnść: 0.8833333333333333
Macież błędów:
 [[24  2]
 [ 5 29]]

Klasyczne drzewo decyzyjne:
Czas budowy: 0.4711151123046875 milisekund
Czas użycia: 0.0667572021484375 milisekund
Dokładność: 0.8
Macież błędów:
 [[22  4]
 [ 8 26]]
~~~
~~~
Zbór danych: iris

Rozmyte drzewo decyzyjne:
Czas budowy: 479.2449474334717 milisekund
Czas użycia: 1.3551712036132812 milisekund
Dokładnść: 1.0
Macież błędów:
 [[10  0  0]
 [ 0  9  0]
 [ 0  0 11]]

Klasyczne drzewo decyzyjne:
Czas budowy: 0.32901763916015625 milisekund
Czas użycia: 0.06198883056640625 milisekund
Dokładność: 1.0
Macież błędów:
 [[10  0  0]
 [ 0  9  0]
 [ 0  0 11]]
~~~
~~~
Zbór danych: moons

Rozmyte drzewo decyzyjne:
Czas budowy: 2392.4028873443604 milisekund
Czas użycia: 2.9571056365966797 milisekund
Dokładnść: 0.8333333333333334
Macież błędów:
 [[20  6]
 [ 4 30]]

Klasyczne drzewo decyzyjne:
Czas budowy: 0.41604042053222656 milisekund
Czas użycia: 0.06389617919921875 milisekund
Dokładność: 0.7333333333333333
Macież błędów:
 [[15 11]
 [ 5 29]]
~~~

## Podsumowanie
