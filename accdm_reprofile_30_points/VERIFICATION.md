# Weryfikacja pakietu

Weryfikacja dotyczy poprawności kodu i przebiegu kampanii. Nie oznacza wykonania nowej kampanii accDM ani usunięcia obserwowanych oscylacji.

Końcowy zestaw: **30 testów, wszystkie zakończone powodzeniem**. Dodatkowo sprawdzono selekcję wyników w raporcie, interfejs CLI oraz składnię notebooka. Log zestawu dołączono jako `verification_test_log.txt`.

## Dane rzeczywiste

W archiwum projektu odzyskano `lyalpha_pt/fit.py` o hash zgodnym z przekazaną tabelą reprodukcji:

`845ec6896c8901df8535d2bc8069344842235e1fc38ba4bca7fd5a59acfb30b6`

Testy integracyjne wykorzystały rzeczywiste archiwalne `lcdm_2022_planck_precision_v3_3.npz` i odpowiadający mu fit JSON wraz z danymi projektu. Odtworzona χ² tego konkretnego zapisu wynosi **193.19661059914839**. Jest to kontrola archiwalnego LCDM, nie wynik nowego dopasowania accDM.

Sprawdzono odtworzenie zapisanej funkcji celu, modelu, reszt i pochodnych, ograniczone profilowanie amplitud oraz wywołania L-BFGS-B, pełnej metody sześciowymiarowej i differential evolution. Przetestowano również 80 ustawień czterech parametrów w rozszerzonym przedziale przy użyciu rzeczywistego fittera. Parametry modyfikowane w tych kontrolach są testami numerycznymi, nie dodatkowymi wynikami kampanii.

## Przypadki numeryczne i obsługa kampanii

- Minimum wewnętrzne, cztery aktywne krawędzie dopuszczalnego obszaru amplitud oraz porównanie z niezależną optymalizacją ograniczoną.
- Skorelowana kowariancja, prawie współliniowe kolumny i utrata rzędu macierzy modelu.
- Zachowanie bardzo małej amplitudy countertermu, w tym wartości rzędu 10⁻⁸⁷, zera i znaku.
- Zgodność analitycznego jakobianu z różnicami skończonymi oraz pełnego wektora reszt z produkcyjną χ².
- Tożsamość zapisywanej χ² i parametrów dla wyniku końcowego, najlepszego odwiedzonego punktu oraz wyniku zachowanego po wyczerpaniu budżetu.
- Odrzucanie niezgodnych parametrów, granic, digestu, trybu kowariancji i niepełnego snapshotu.
- Zatrzymanie kampanii, gdy walidacja reszt lub jakobianu zwróci `passed=False`.
- Pełny techniczny przebieg pilota, 30 punktów i wznowienia, bez duplikowania zakończonych prób. W tym teście fitter był celowo zastąpiony prostą funkcją testową; nie są to wyniki accDM.
- Stabilne przypisanie startów do prób także po przerwaniu pomiędzy kontrolami pełnej metody sześciowymiarowej.

Osobny test raportowania potwierdził, że porównanie kierunków korzysta z końcowych wyników prób zgłaszających sukces, a nie z lepszego punktu zachowanego ze startu lub wcześniejszej iteracji. Raport nie wymaga zmiany wektora jako warunku zbieżności. Przy braku odpowiednich prób nie pokazuje nieaktualnego wykresu porównawczego.

Sprawdzono składnię modułów i wszystkich komórek notebooka. Notebook celowo nie zawiera wykonanych komórek z pozorowanymi wynikami kampanii.

Środowisko weryfikacji: Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, pandas 2.2.3, matplotlib 3.10.8. Pakiet korzysta ze środowiska i danych istniejącego projektu użytkownika; walidacja uruchomieniowa jest częścią każdej nowej kampanii.

## Co pozostaje do sprawdzenia na Twoich danych

Pierwszym rzeczywistym uruchomieniem jest czteropunktowy pilot. Dopiero jego wyniki pokażą, czy znane poprawki zostały odzyskane w środowisku produkcyjnym, a pełna kampania pozwoli ocenić główne oscylacje. Zgodność metod zwiększa wiarygodność minimum, lecz nie stanowi dowodu globalnej optymalności. Nowy profiler nie sprawdza przez ponowne obliczenie dokładności widm CLASS ani całek jednoloopowych.
