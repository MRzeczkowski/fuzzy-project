# Porównanie rozmytego i klasycznego drzewa decyzyjnego

## Wstęp

Celem tego projektu było porównanie wybranych parametrów klasyfikacji za pomocą rozmytego i klasycznego drzewa decyzyjnego. W tym celu wykorzystano dziesięć zbiorów danych pobranych ze strony [Kaggle](https://www.kaggle.com/), za pomocą bibliotek [fuzzy-tree](https://balins.github.io/fuzzytree/index.html) i [Sklearn](https://scikit-learn.org/stable/) zbudowano odpowiednio rozmyte i klasyczne drzewo decyzyjne oraz za pomocą skryptu w języku Python wygenerowano raport z wybranymi statystykami.


## Opis zbiorów danych


## Algorytm budowania rozmytego drzewa decyzyjnego


## Raport
Poniższy Raport zawiera dla każdego zbioru danych informacje o tym jak długo trwało budowanie drzew, ich użycie, dokładność klasyfikacji na zbiorze testującym oraz macież błędów która może zostać użyta do dalszej analizy.

~~~
Zbór danych: animals

Klasyczne drzewo decyzyjne:
Czas budowy: 1.9650459289550781 milisekund
Czas użycia: 0.1270771026611328 milisekund
Dokładność: 0.9523809523809523
Precyzja: 0.7142857142857143
Czułość: 0.7142857142857143
Macież błędów:
 [[12  0  0  0  0  0  0]
 [ 0  2  0  0  0  0  0]
 [ 0  0  0  0  1  0  0]
 [ 0  0  0  2  0  0  0]
 [ 0  0  0  0  0  0  0]
 [ 0  0  0  0  0  3  0]
 [ 0  0  0  0  0  0  1]]

Rozmyte drzewo decyzyjne:
Czas budowy: 14.976978302001953 milisekund
Czas użycia: 0.25200843811035156 milisekund
Dokładnść: 0.9523809523809523
Precyzja: 0.7142857142857143
Czułość: 0.7142857142857143
Macież błędów:
 [[12  0  0  0  0  0  0]
 [ 0  2  0  0  0  0  0]
 [ 0  0  0  0  1  0  0]
 [ 0  0  0  2  0  0  0]
 [ 0  0  0  0  0  0  0]
 [ 0  0  0  0  0  3  0]
 [ 0  0  0  0  0  0  1]]
~~~
~~~
Zbór danych: fetal_health

Klasyczne drzewo decyzyjne:
Czas budowy: 8.166313171386719 milisekund
Czas użycia: 0.14472007751464844 milisekund
Dokładność: 0.9225352112676056
Precyzja: 0.8712892789079466
Czułość: 0.8954924752338544
Macież błędów:
 [[314  17   2]
 [ 11  52   1]
 [  2   0  27]]

Rozmyte drzewo decyzyjne:
Czas budowy: 119508.06498527527 milisekund
Czas użycia: 29.098033905029297 milisekund
Dokładnść: 0.9248826291079812
Precyzja: 0.9214287001495739
Czułość: 0.8313550619585102
Macież błędów:
 [[326   7   0]
 [ 20  44   0]
 [  3   2  24]]
~~~
~~~
Zbór danych: glass

Klasyczne drzewo decyzyjne:
Czas budowy: 0.7660388946533203 milisekund
Czas użycia: 0.09512901306152344 milisekund
Dokładność: 0.7906976744186046
Precyzja: 0.8088624338624338
Czułość: 0.8003246753246754
Macież błędów:
 [[10  0  0  0  0  1]
 [ 4  9  1  0  0  0]
 [ 0  0  3  0  0  0]
 [ 0  1  0  1  2  0]
 [ 0  0  0  0  3  0]
 [ 0  0  0  0  0  8]]

Rozmyte drzewo decyzyjne:
Czas budowy: 3809.9892139434814 milisekund
Czas użycia: 2.285003662109375 milisekund
Dokładnść: 0.7674418604651163
Precyzja: 0.6648809523809524
Czułość: 0.6644119769119768
Macież błędów:
 [[10  1  0  0  0  0]
 [ 3 11  0  0  0  0]
 [ 1  2  0  0  0  0]
 [ 0  1  0  3  0  0]
 [ 0  0  0  0  2  1]
 [ 0  0  0  0  1  7]]
~~~
~~~
Zbór danych: heart_attack

Klasyczne drzewo decyzyjne:
Czas budowy: 0.6051063537597656 milisekund
Czas użycia: 0.06198883056640625 milisekund
Dokładność: 0.819672131147541
Precyzja: 0.826797385620915
Czułość: 0.8232758620689655
Macież błędów:
 [[26  3]
 [ 8 24]]

Rozmyte drzewo decyzyjne:
Czas budowy: 2248.9311695098877 milisekund
Czas użycia: 2.8548240661621094 milisekund
Dokładnść: 0.819672131147541
Precyzja: 0.8193548387096774
Czułość: 0.8200431034482758
Macież błędów:
 [[24  5]
 [ 6 26]]
~~~
~~~
Zbór danych: mobile_price_train

Klasyczne drzewo decyzyjne:
Czas budowy: 7.922887802124023 milisekund
Czas użycia: 0.1800060272216797 milisekund
Dokładność: 0.83
Precyzja: 0.8290811191626408
Czułość: 0.8257286192068801
Macież błędów:
 [[ 90  15   0   0]
 [  6  79   6   0]
 [  0  14  63  15]
 [  0   0  12 100]]

Rozmyte drzewo decyzyjne:
Czas budowy: 242738.862991333 milisekund
Czas użycia: 39.1690731048584 milisekund
Dokładnść: 0.9125
Precyzja: 0.9110206137010463
Czułość: 0.9124049410734194
Macież błędów:
 [[ 98   7   0   0]
 [  3  85   3   0]
 [  0   8  81   3]
 [  0   0  11 101]]
~~~

## Podsumowanie
