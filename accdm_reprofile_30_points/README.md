# Nowe profilowanie accDM: 30 zapisanych punktów

Pakiet ponownie dopasowuje sześć parametrów nuisance do istniejących widm. Najpierw uruchamia **pilot czterech punktów**, następnie właściwą kampanię 30 punktów. Nie wykonuje nowych obliczeń CLASS ani całek jednoloopowych. Wyniki trafiają do osobnego katalogu; produkcyjne JSON/NPZ pozostają wejściami tylko do odczytu.

**Status:** kod sprawdzono na rzeczywistym archiwalnym LCDM oraz na testach numerycznych i technicznych. Nie uruchomiono nim produkcyjnych 30 punktów accDM, ponieważ ich pliki NPZ i fit JSON znajdują się w Twoim projekcie. W paczce jest pełny indeks 1600 zapisanych punktów i manifest wyboru, ale nie ma tych produkcyjnych widm.

## Uruchomienie

Rozpakuj ZIP w głównym katalogu `lyalpha_one_loop`. Powstanie podkatalog `accdm_reprofile_30_points`. Użyj środowiska Pythona, w którym działa Twój fitter. Potrzebne są jego dotychczasowe zależności oraz NumPy, SciPy, pandas i matplotlib; do notebooka także Jupyter/IPython. Nie trzeba zastępować modułów projektu ani instalować innej wersji fittera.

Najwygodniej otworzyć **11_reprofile_accdm_30_points.ipynb**, sprawdzić `PROJECT_ROOT` i uruchomić komórki. Domyślna faza to `pilot`.

Alternatywnie, z głównego katalogu projektu:

```bash
python accdm_reprofile_30_points/run_reprofile.py \
  --project-root "$PWD" \
  --phase pilot \
  --out runs/accdm/reprofile_30_v1
```

Po pomyślnym pilocie:

```bash
python accdm_reprofile_30_points/run_reprofile.py \
  --project-root "$PWD" \
  --phase campaign \
  --out runs/accdm/reprofile_30_v1 \
  --resume
```

Przerwany pilot wznawiaj tą samą komendą pilota z `--resume`. Przerwaną kampanię wznawiaj drugą komendą. Punkt kontrolny zapisuje się po każdej zakończonej próbie optymalizacji; próba przerwana w trakcie będzie powtórzona. Nie uruchamiaj jednocześnie dwóch procesów zapisujących ten sam katalog.

Same raporty, bez dopasowania:

```bash
python accdm_reprofile_30_points/run_reprofile.py \
  --phase report --out runs/accdm/reprofile_30_v1
```

Opcjonalna faza `validate` sprawdza wszystkie 30 wejść bez optymalizacji. Po jej użyciu dalsze fazy wymagają `--resume`. W notebooku fazę `campaign` ustaw ręcznie; domyślne „Run All” nie uruchamia całej kampanii.

## Co jest sprawdzane przed dopasowaniem

- Odtworzenie zapisanej χ² przez produkcyjny fitter, zgodność sześciu parametrów i ich dokładnych ograniczeń.
- Tożsamość widma, danych, kowariancji i wersji kodu; zakresy k, redshifty oraz tryb kowariancji.
- Zgodność analitycznego modelu profilowania i pełnego wektora reszt z funkcją produkcyjną; numeryczna kontrola pochodnych.
- Przy wznawianiu: niezmienność kodu, manifestu, indeksu danych, konfiguracji i zweryfikowanych plików.

Obsługiwana wersja `lyalpha_pt/fit.py` ma SHA256:

`845ec6896c8901df8535d2bc8069344842235e1fc38ba4bca7fd5a59acfb30b6`

Jest to dokładny hash fittera z przekazanej tabeli reprodukcji. Odzyskana kopia występowała w archiwum `lyalpha_one_loop_campaigns_v3.6.0`. Jeśli hash lokalnego pliku jest inny, program zatrzyma się: trzeba sprawdzić zgodność modelu, a nie wyłączać kontrolę.

Ścieżki zapisane względem znanych katalogów projektu na klastrze `/home/2/ks405818/Master/lyalpha_one_loop` i lokalnie `/home/krzysztof/Workspace/lyalpha_one_loop` są mapowane na `--project-root`. Ścieżki względne również odnoszą się do tego katalogu. Inne ścieżki bezwzględne pozostają jawne. Dostępność plików jest sprawdzana na komputerze uruchamiającym program.

Domyślny snapshot jest zamrożony. `--snapshot` pozwala wskazać zgodny pełny eksport, zawierający wszystkie 1600 współrzędnych oraz wymagane kolumny. Zmiana snapshotu lub kodu wymaga nowego katalogu wyników. Nie używaj podzbioru 30 punktów jako snapshotu, bo zmieniłoby to odniesienie χ² i pulę startów sąsiednich.

## Strategia obliczeń

1. **Pilot: cztery punkty.** Dwa wykryte błędne profile `(14.7142857143, −1.375)` i `(15, −1.375)`, kontrola `(19.2857142857, −4)` i punkt oscylacji `(14.7142857143, −0.1)`. Każdy dostaje do dwóch startów lokalnych i niezależną próbę pełnego dopasowania sześciu parametrów. Bramka wymaga m.in. zachowania znanych kandydatów χ² ≤192.053545084295 i ≤193.904599510029, z tolerancją 0.001. To kontrola poprawności i odzyskania znanych poprawek, nie dowód globalnego minimum.
2. **Starty globalne:** differential evolution, po dwa ustalone ziarna w trzech punktach kotwiczących: `(16, 0)`, `(16.4285714286, −0.2)`, `(14.7142857143, −0.1)`.
3. **Przejścia w obu kierunkach:** lokalna optymalizacja czterech parametrów po profilowaniu dwóch amplitud; starty własne, sąsiednie i z poprzedniego punktu. Oba kierunki korzystają ze wspólnej puli zamrożonej przed przejściami. Każdy punkt ma do dwóch startów na kierunek; przecięcia przekrojów są wykonywane raz na kierunek i później używane ponownie.
4. **Kontrola inną metodą:** pełne sześć parametrów przez `least_squares`, z analitycznym jakobianem, do dwóch różnych najlepszych startów na punkt.

Wszystkie próby używają tych samych zapisanych widm, danych, kowariancji i ograniczeń nuisance. Bieżące dopasowanie nie poszerza granic. Test wpływu granic będzie osobnym eksperymentem, jeśli nadal zobaczymy minima na krawędzi.

Zamiast normalizować małą `alpha_ct` przez środek szerokiego przedziału, profilujemy amplitudy `a = exp(log_alpha_F)` i `c = a * alpha_ct`. Dla ustalonych pozostałych czterech parametrów model jest liniowy w `(a,c)`. Minimum jest szukane we wnętrzu i na krawędziach dokładnego dozwolonego obszaru: `a_min ≤ a ≤ a_max`, `alpha_ct_min*a ≤ c ≤ alpha_ct_max*a`. Uwzględnienie krawędzi jest istotne: stary profiler odrzucał niedozwolone minimum bezwarunkowe, zamiast znajdować minimum ograniczone. Obliczenia korzystają z wybielania kowariancją i stabilnego skalowania kolumn. Pełna metoda sześciowymiarowa pracuje na fizycznych parametrach i zachowuje amplitudy rzędu 10⁻⁸⁷.

Domyślne budżety to `--local-budget 5000` i `--global-budget 20000`. Budżet obejmuje wywołania funkcji przez solver, a w metodzie sześciowymiarowej również jakobianu; nie obejmuje wszystkich wewnętrznych walidacji i ponownych ocen χ². Ziarna DE: `12345 23456`. Zmiana budżetu lub ziaren wymaga osobnego katalogu wyników. Budżet nie jest gwarancją zbieżności ani czasu wykonania.

## Dokładny wybór punktów

Manifest `pilot_points.csv` zawiera **30 unikalnych punktów**:

| Grupa | Liczba | Cel |
|---|---:|---|
| Przekroje głównych oscylacji | 22 | dwa przekroje po masie i dwa po frakcji |
| Okolica dwóch poprawionych profili | 3 | odzyskanie znanego ulepszenia i kontrola sąsiada |
| Prostokąt przy Δχ²≈5.991 | 4 | rozdzielenie wpływu minimum i ograniczenia `beta_bias` |
| Kontrola przy małym f | 1 | stabilny obszar porównawczy |

Nie ma zapisanego punktu `(14.7142857143, −0.3)` w tym przekroju. Raport zachowuje tę przerwę. Braki nie są uzupełniane przez interpolację.

## Wyniki do oceny

| Plik | Znaczenie |
|---|---|
| `reprofile_best_points.csv` | χ² przed/po, poprawa, sześć parametrów, ograniczenia, status stabilności |
| `reprofile_attempts.csv` | każda próba: start, koniec solvera, najlepszy odwiedzony i zachowany wektor, budżet, status |
| `reprofile_p1d.csv` | przewidywane P1D przed/po na siatce danych, dane i błędy diagonalne |
| `reprofile_chi2_by_redshift.csv` | produkcyjne rozbicie χ² po redshiftach |
| `validation.json` | odtworzenie wyniku i identyfikatory wejść, parametrów numerycznych i kontroli modelu |
| `pilot_gate.json` | wynik czteropunktowej kontroli wstępnej |
| `campaign_metadata.json` | wspólna referencja, konfiguracja i zakres wykonanych punktów |
| `state.json` | stan umożliwiający wznowienie |
| `reports/` | przekroje χ², parametry nuisance, granice, P1D i porównania kierunków |

Wspólne odniesienie obu wersji profilu to minimum pełnego snapshotu: około **188.429597466013**. Poprawa punktów nie przesuwa automatycznie zera wykresu. Progi 2.30 i 5.991 służą tu do porównania z dotychczasowym wykresem, nie do ponownego uzasadnienia ich interpretacji statystycznej.

Najważniejsze kolumny diagnostyczne:

- `gain = chi2_old − chi2_best`: poprawa najlepszego **jawnie ocenionego** wektora.
- `termination = budget_exhausted`: budżet wyczerpany; zachowany kandydat nadal może być lepszy, ale próba nie potwierdza zbieżności.
- `chi2_optimizer_final` i `optimizer_success`: wynik zwrócony przez solver; odrębny od `chi2_selected`, które może pochodzić ze startu lub wcześniejszej iteracji.
- `stability_assessment = agreement_observed_not_global_proof`: oba kierunki i pełna metoda sześciowymiarowa zgłosiły sukces, a ich końcowe χ² zgadzają się z najlepszym wynikiem do 0.01; przy Δχ² odległym od 5.991 o mniej niż 0.05 tolerancja wynosi 0.003. Są to robocze kryteria diagnostyczne.
- `further_search_needed`: brakuje powyższej zgodności. Sam spadek χ² lub zachowanie starego minimum nie oznacza rozwiązania problemu oscylacji.

Po pilocie obejrzyj odzyskanie dwóch znanych poprawek i status punktu oscylacji. Po kampanii interesują nas przede wszystkim: zmiany czterech przekrojów χ², zgodność obu kierunków i pełnej metody, aktywne granice nuisance oraz różnice P1D. Jeśli sprawdzone minima nadal odtwarzają oscylacje, następnym testem będzie precyzja teorii/projekcji P1D dla wybranych sąsiadów. Ta kampania sama nie potwierdza dokładności zapisanych widm.

Notebook na końcu tworzy ZIP z aktualnymi tabelami, metadanymi i raportami. Ten ZIP wystarczy przekazać do dalszej analizy; nie trzeba ponownie wysyłać wszystkich rysunków osobno.
