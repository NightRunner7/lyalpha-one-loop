# Ly-alpha one-loop: historia poprawek i instrukcja użytkownika

## 1. Najkrótszy wniosek

Obecny projekt rozdziela obliczenie na trzy niezależne warstwy:

1. **model i CLASS** -- tylko `ModelSpec` oraz surowe parametry wejściowe CLASS;
2. **generator teorii** -- widma liniowe, `P22`, `P13` i trzy kanały
   `P_dd`, `P_dtheta`, `P_thetatheta`, zapisane w jednym pliku `.npz`;
3. **projekcja P1D i fit** -- działa lokalnie z pliku `.npz`, bez CLASS.

Fit jest niezależny od nazwy modelu i jego parametrów kosmologicznych. Musi
jedynie otrzymać poprawny `TheoryBundle` na siedmiu redshiftach. Nie oznacza to
jednak, że zastosowanie standardowych jąder EdS do dowolnej nowej fizyki jest
automatycznie poprawnym przybliżeniem. Wybór widma źródłowego pętli i jej wagi
jest częścią definicji fizycznej modelu i musi zostać uzasadniony.

## 2. Co było nie tak w starszych wersjach

### 2.0. Najwcześniejszy uproszczony one-loop

W pierwszych próbach nie zawsze konstruowaliśmy osobno trzech nieliniowych
kanałów `P_dd`, `P_dtheta` i `P_thetatheta`. Użycie jednego widma materii w
całej strukturze kątowej jest poprawnym skrótem na poziomie liniowym, gdzie
trzy kanały są równe, ale nie jest pełną implementacją one-loop z równań pracy.
Taki model może dać wizualnie dobry fit dzięki nuisance parameters, lecz nie
testuje właściwej korekty prędkościowej.

**Zmiana:** generator zawsze zapisuje oddzielne `dd`, `dtheta` i `thetatheta`,
a projekcja używa ich odpowiednio w `I0`, `I2` i `I4`. Każdy kanał ma osobne
`P22` i `P13` oraz jest kontrolowany przed wykonaniem likelihoodu.

### 2.1. Monolityczna architektura

Pierwotny plik łączył konfigurację CLASS, liczenie pętli, projekcję 3D do 1D,
model efektywny, kowariancję oraz minimalizację. Samo połączenie tych elementów
nie zmienia równań, ale uniemożliwiało łatwe rozstrzygnięcie, czy zły wynik
pochodzi z teorii, projekcji, danych czy optymalizatora. Fit lokalny uruchamiał
również CLASS, mimo że po wygenerowaniu teorii nie było to potrzebne.

**Zmiana:** `models.py`, `theory.py` i `fit.py` są niezależne. Gotowy bundle
zawiera wszystko, czego potrzebuje projekcja i likelihood.

### 2.2. Niebezpieczny cache

Nazwa starego cache zależała między innymi od redshiftu, końca siatki i kilku
liczb punktów. Nie zawierała jednak pełnej kosmologii, wszystkich parametrów
CLASS, rzeczywistego widma liniowego, regulatora `P13` ani wersji algorytmu.
Po zmianie LCDM na DCDM lub accDM istniało więc realne ryzyko wczytania
wcześniejszych pętli.

**Zmiana:** klucz checkpointu jest skrótem SHA-256 obejmującym:

- wersję generatora;
- kompletny `ModelSpec` i słownik wejściowy CLASS;
- wszystkie ustawienia numeryczne;
- redshift i siatkę `k`;
- rzeczywiste widma `P_total` oraz widmo użyte do pętli.

Końcowy bundle ma osobny digest obejmujący wszystkie zapisane tablice. Digest
jest sprawdzany przy każdym wczytaniu, a fit JSON zapisuje digest teorii.

### 2.3. Preskrypcja neutrinowa

W jednej z wcześniejszych wersji można było wybrać pętlę z `P_cb`, ale brakowało
czynnika `(1-f_nu)^2`. Domyślnie używane było też widmo całkowite, niezależnie
od benchmarku.

**Zmiana:** model jawnie definiuje dwie oddzielne rzeczy:

```text
loop_source = total albo cb
loop_weight = liczba albo one_minus_fnu_squared
```

Dla benchmarku z masywnym neutrinem:

```text
P_XY = P_total_linear + (1-f_nu)^2 * (P22_XY[P_cb] + P13_XY[P_cb]).
```

Tree level pozostaje widmem całkowitej materii.

### 2.4. Jeden parametr `k_max` pełnił kilka ról

W monolitycznej wersji `k_max` był jednocześnie końcem obliczonej teorii i
górnym limitem projekcji P1D. W jeszcze wcześniejszych eksperymentalnych
notebookach pojawiał się taper między `k_trust` i `k_end`, po którym poprawkę
one-loop zastępowano wynikiem liniowym. Taki zabieg nie odtwarza formalizmu
publikacji.

**Zmiana:** obecnie rozróżniamy:

| wielkość | znaczenie |
| --- | --- |
| `k_output_max` | koniec zapisanych kanałów one-loop, standardowo 20 h/Mpc |
| `q_max` | wewnętrzny cutoff całek SPT, 50 lub 100 h/Mpc |
| `k_trust` | zakres niezależnej kontroli surowego SPT, standardowo 2 h/Mpc |
| `k_uv_cut` | techniczny górny limit projekcji P1D, skanowany 10--20 h/Mpc |

`k_trust` nie wyłącza pętli. W obecnym modelu nie ma taperu ani powrotu do
widma liniowego na części siatki.

### 2.5. Regulator i kwadratura `P13`

Konfiguracja `(k,q,-q)` jest osobliwa dla bezpośredniej rekurencji `F3/G3`.
Stary generator rozsuwał wektory regulatorem `epsilon=10^-4` i całkował
dwuwymiarowo. Wynik zależał od regulatora oraz rozdzielczości. Ponieważ `P22`
i `P13` silnie się kasują, niewielki błąd pojedynczego składnika stawał się
procentowym błędem w ich małej sumie.

**Zmiana:** produkcja używa zamkniętych, jednowymiarowych postaci EdS dla
wszystkich trzech kanałów `P13`. Bezpośrednia symetryzowana rekurencja z
ekstrapolacją Richardsona pozostała niezależnym testem, a nie metodą
produkcyjną. Bundle zapisuje także oszacowanie błędu rozdzielczości `P13`.

### 2.6. Brzeg `p -> 0` w `P22`

Zmienne `(q,p=|k-q|)` były już lepsze od bezpośredniej całki po kącie, ale
pełna domena nadal zawierała nieanalityczny brzeg `p -> 0` przy `q=k`. Wynik
wykazywał efekt parzystej/nieparzystej liczby punktów. Przy dużym kasowaniu
`P22+P13` błąd był mocno wzmacniany.

**Zmiana:** całka korzysta z dokładnej symetrii `q <-> p`, liczy tylko połowę
domeny `p>=q` i mnoży wynik przez dwa. Domena jest dzielona przy `q=k/2`, gdzie
zmienia się jej dolna granica. Osobna, wolniejsza całka po pełnej domenie służy
jako test referencyjny.

### 2.7. Projekcja 3D do 1D

Nie znaleziono brakującego czynnika ani błędnej struktury końcowego modelu P1D
w późnej wersji monolitycznej. Problemem było raczej sprzężenie projekcji z
jedną wspólną siatką i znaczeniem `k_max`. W niektórych wcześniejszych testach
niezależne całki dla każdego punktu mogły także generować nieaddytywne różnice
numeryczne przy zmianie cutoffu.

**Zmiana:** `I0`, `I2` i `I4` są liczone jako skumulowane całki trapezowe na
jednej gęstej siatce. Do siatki jawnie wstawiane są wszystkie obserwowane
`k_parallel` i punkt cutoffu. Zmiana `k_uv_cut` zmienia tylko górny limit tej
samej całki. Bundle przechowuje `H(z)` i współczynnik konwersji jednostek.

### 2.8. Dane i kowariancja

W czasie rozwoju kodu używane były różne warianty: sama statystyka, macierz
korelacji, błędy systematyczne oraz dawne pliki z odwrotną kowariancją. To
zmieniało absolutną skalę chi-kwadrat i utrudniało porównanie z publikacją.

**Zmiana:** loader rygorystycznie wymaga publicznych plików o kształtach
`455x6`, `455x8` i trzynastu bloków `35x35`. Fiducjalna selekcja zawsze daje
245 punktów w siedmiu redshiftach `z=3.0,...,4.2`. Benchmarki używają
`paper_diag`, czyli błędów statystycznych i ośmiu systematycznych dodanych
diagonalnie w kwadraturze. Inne warianty są dostępne, ale muszą być wybierane
jawnie i nie powinny być mieszane w jednym porównaniu.

### 2.9. Najważniejszy problem: stare minimum fitu

Po poprawieniu surowego SPT nadal otrzymywaliśmy wyniki około 199 przy
`k_uv=10` i 214--233 przy `k_uv=15--20`. Nie oznaczało to już złej teorii.
Likelihood ma wąskie, rozłączne doliny związane z amplitudą i countertermem.
Stara procedura:

- przeszukiwała od początku pełne sześciowymiarowe i bardzo szerokie pudełko;
- rozcieńczała populację globalną po rozszerzeniu countertermu;
- mogła zaakceptować wynik Powella gorszy od punktu startowego;
- próbowała znaleźć minimum przy 20 h/Mpc bez wykorzystania sąsiedniego
  rozwiązania przy 10 lub 15 h/Mpc.

**Zmiana:** dla ustalonych `(beta_F, alpha_bias, beta_bias, beta_ct)` model jest
liniowy w dwóch kombinacjach:

```text
a = alpha_F
c = alpha_F * alpha_ct.
```

Są one dokładnie profilowane przez rozwiązanie układu `2x2`. Globalne
przeszukiwanie ma więc cztery, a nie sześć wymiarów. Dodatkowo:

- przeszukiwane są wnętrze oraz obie powierzchnie graniczne `beta_ct`;
- używa się kilku seedów;
- każdy kandydat jest poprawiany metodami L-BFGS-B, Neldera-Meada i Powella;
- zawsze zachowywany jest kandydat o najmniejszym chi-kwadrat;
- szeroki box countertermu jest osiągany lokalną kontynuacją z bazowego boxu;
- cutoff 20 jest osiągany ścieżką `10 -> 15 -> 20`.

To właśnie ta zmiana odzyskała wąską dolinę i obniżyła wynik do około 193.
Opcja `--full-six-dimensional` pozostaje wyłącznie cross-checkiem starej
parametryzacji.

## 3. Dlaczego ufamy obecnej wersji

Obecna wersja przechodzi niezależne testy na czterech poziomach:

1. testy algebraiczne jąder i tożsamości między kanałami;
2. pełna domena `P22` kontra produkcyjna połowa domeny;
3. zamknięte `P13` kontra rekurencja `F3/G3` z ekstrapolacją;
4. suma `P22+P13` kontra niezależny FAST-PT/FFTLog.

Dla benchmarku LCDM 2022 maksymalna różnica z FAST-PT na `k<=2 h/Mpc` wynosi
około `6.1e-4` widma tree. Pełny łańcuch daje:

```text
chi2_linear   = 206.248063
chi2_one_loop = 193.196611
Delta chi2    = -13.051453
```

wobec około `206.29`, `192.89` i `-13.4` w arXiv:2210.06117.

## 4. Co dokładnie jest model-independent

### Niezależne od modelu

- wczytanie danych;
- konstrukcja kowariancji;
- konwersja i projekcja P1D z zapisanego bundle'a;
- SiIII, wygładzenie termiczne i filtr `k_F`;
- sześcioparametrowy model nuisance;
- optymalizator, cutoff scan, raportowanie i wykresy.

### Zależne od modelu

- słownik wejściowy CLASS;
- źródło liniowego widma do pętli: `total` lub `cb`;
- waga pętli;
- zasadność użycia standardowych jąder EdS.

Generator jest technicznie zgodny z dowolnym wrapperem CLASS, który udostępnia
`pk_lin`, `Hubble`, `h`, `Omega_m`, `sigma8` oraz -- dla `loop_source=cb` --
`pk_cb_lin`. Dla modyfikowanej grawitacji, silnie wielopłynowej dynamiki lub
modelu zmieniającego nieliniowe jądra zastosowanie EdS SPT może być tylko
przybliżeniem. Tego nie rozstrzyga dobry fit nuisance.

## 5. Instalacja

### Fit i notebook na komputerze lokalnym

```bash
cd /sciezka/do/lyalpha_one_loop
python -m pip install -e '.[notebook]'
python -m unittest discover -s tests -v
```

Instalacja `-e` jest zalecana: import `lyalpha_pt` wskazuje wtedy zawsze na
bieżący katalog projektu. CLASS nie jest potrzebny do fitu gotowego `.npz`.

Do niezależnego testu FAST-PT:

```bash
python -m pip install -e '.[validation]'
```

### Generator na klastrze

Aktywuj środowisko zawierające właściwy, już działający wrapper CLASS, a potem:

```bash
cd /sciezka/do/lyalpha_one_loop
python -m pip install -e .
python -c "from classy import Class; print('CLASS import OK')"
```

Nie instaluj stockowego `classy`, jeśli DCDM lub accDM wymaga zmodyfikowanej
gałęzi CLASS. Polecenie `pip install classy` może zastąpić właściwy wrapper.

## 6. Definicja nowego modelu

Najwygodniej skopiować `examples/custom_lcdm_model.json`:

```bash
cp examples/custom_lcdm_model.json examples/my_model.json
```

Minimalny schemat:

```json
{
  "name": "unikalna_nazwa_modelu",
  "description": "Opis punktu i konwencji parametrów",
  "class_params": {
    "omega_b": 0.02237,
    "omega_cdm": 0.1200,
    "100*theta_s": 1.04110,
    "ln10^{10}A_s": 3.044,
    "n_s": 0.9649,
    "tau_reio": 0.0544,
    "N_ur": 2.0328,
    "N_ncdm": 1,
    "m_ncdm": 0.06
  },
  "loop_source": "cb",
  "loop_weight": "one_minus_fnu_squared",
  "tags": ["custom"]
}
```

Zasady:

1. `class_params` zawiera surowe parametry przyjmowane przez `Class.set()`, a
   nie nazwy parametrów MontePython po skalowaniu lub logarytmowaniu.
2. Nazwa modelu i nazwy plików powinny jednoznacznie identyfikować punkt.
3. Nie pozostawiaj jednocześnie standardowego `omega_cdm` i parametru, który
   zastępuje go w niestandardowej gałęzi.
4. Dla DCDM zastąp `omega_cdm` początkową gęstością DCDM wymaganą przez daną
   gałąź. Dokładna nazwa (`omega_ini_dcdm`, `omega_ini_dcdm2` lub inna) zależy
   od wrappera i musi pochodzić z działającego inputu tej gałęzi.
5. Parametry takie jak `Log10_Gamma_dcdm` używane przez MontePython mogą być
   tylko parametrami próbkowania. Do JSON-u wpisz końcową wartość i nazwę,
   której oczekuje bezpośrednio CLASS.
6. Dla accDM obowiązuje ta sama reguła: zacznij od działającego wywołania
   `Class.set()`, a nie od pliku `.param` MontePython bez sprawdzenia konwersji.

### Wybór pętli

```json
"loop_source": "total",
"loop_weight": 1.0
```

oznacza bezpośrednią pętlę EdS z całkowitego liniowego widma materii.

```json
"loop_source": "cb",
"loop_weight": "one_minus_fnu_squared"
```

oznacza preskrypcję z masywnymi neutrinami zastosowaną w benchmarku.
Nie wybieraj wariantu na podstawie tego, który daje niższe chi-kwadrat.

## 7. Generowanie teorii

### Krok 1: smoke

```bash
python scripts/generate_theory.py \
  --model-json examples/my_model.json \
  --quality smoke \
  --output results/my_model_smoke.npz \
  --checkpoint-dir results/checkpoints_my_model_smoke
```

Smoke sprawdza CLASS, wejście/wyjście, kanały i zapis. Ma tylko 28 punktów `k`
i nie jest wynikiem naukowym.

### Krok 2: production

```bash
python scripts/generate_theory.py \
  --model-json examples/my_model.json \
  --quality production \
  --output results/my_model_production.npz \
  --checkpoint-dir results/checkpoints_my_model_production
```

### Krok 3: precision

```bash
python scripts/generate_theory.py \
  --model-json examples/my_model.json \
  --quality precision \
  --output results/my_model_precision.npz \
  --checkpoint-dir results/checkpoints_my_model_precision
```

Profile numeryczne:

| profil | punkty wyjściowe k | q_max | przeznaczenie |
| --- | ---: | ---: | --- |
| smoke | 28 | 20 | tylko test programu |
| production | 110 | 50 | pierwszy wynik fizyczny |
| precision | 160 | 100 | wynik referencyjny |

Jeżeli job zostanie przerwany, uruchom dokładnie tę samą komendę. Ukończone
redshifty zostaną wczytane wyłącznie wtedy, gdy cały hash się zgadza.
`--overwrite-checkpoints` stosuj tylko wtedy, gdy celowo chcesz przeliczyć
pasujące checkpointy.

Szablon PBS dla JSON-u to `scripts/run_theory_pbs_model_json.sh`.

## 8. Kontrola bundle'a i surowej teorii

Podstawowe podsumowanie:

```bash
python scripts/inspect_results.py \
  --theory results/my_model_precision.npz
```

Porównanie production i precision na zakresie kontrolnym:

```bash
python scripts/compare_theories.py \
  results/my_model_production.npz \
  results/my_model_precision.npz \
  --k-max 2
```

Niezależna kontrola całek:

```bash
python scripts/validate_raw_spt.py \
  --theory results/my_model_precision.npz \
  --redshift 3 \
  --output-dir results/my_model_raw_validation
```

Bez zainstalowanego FAST-PT można dodać `--skip-fastpt`. Test nadal porówna
produkcję z bezpośrednimi kwadraturami `P22` i rekurencją `P13`.

Nie ma jednego uniwersalnego progu dla dowolnej nowej fizyki. Jako praktyczny
cel numeryczny można wymagać, aby production i precision różniły się na
`k<=2 h/Mpc` znacznie mniej niż efekt modelu, typowo nie więcej niż około
`10^-3` w względnym kanale. Tolerancję należy ustalić przed skanem parametrów.

## 9. Fit P1D

Zalecana komenda:

```bash
python scripts/fit_p1d.py \
  --theory results/my_model_precision.npz \
  --data-dir data \
  --mode both \
  --k-uv-cut 20 \
  --covariance paper_diag \
  --seeds 12345 23456 34567 45678 56789 \
  --expanded-counterterm \
  --output results/my_model_fit.json
```

Ta komenda:

- dopasowuje linear i one-loop;
- profiluje dwie amplitudy i globalnie przeszukuje cztery parametry;
- rozpoczyna od bazowego boxu;
- rozszerza counterterm wyłącznie przez kontynuację;
- dochodzi do cutoffu 20 ścieżką `10 -> 15 -> 20`;
- zapisuje wszystkie parametry, wkłady redshiftów, kandydatów i odległości od
  granic.

Opcji `--no-cutoff-continuation` nie używaj w wyniku fiducjalnym. Jest ona
diagnostyką pokazującą, jak łatwo bezpośredni fit przy 20 może ominąć minimum.

Opcja `--full-six-dimensional` wyłącza profilowanie. Powinna służyć tylko jako
kosztowny cross-check, a nie domyślna metoda.

## 10. Kontrola stabilności fitu

### 10.1. Podsumowanie i zgodność digestów

```bash
python scripts/inspect_results.py \
  --theory results/my_model_precision.npz \
  --fit results/my_model_fit.json
```

Skrypt przerywa pracę, jeżeli JSON fitu pochodzi z innego bundle'a.

### 10.2. Skan cutoffu

```bash
python scripts/scan_cutoffs.py \
  --theory results/my_model_precision.npz \
  --data-dir data \
  --cutoffs 10 15 20 \
  --seeds 12345 23456 34567 45678 56789 \
  --expanded-counterterm \
  --output results/my_model_cutoff_scan.json
```

Następnie:

```bash
python scripts/inspect_results.py \
  --theory results/my_model_precision.npz \
  --fit results/my_model_fit.json \
  --cutoff-scan results/my_model_cutoff_scan.json
```

Parametry nuisance mogą silnie zmieniać się z cutoffem. Ważniejsze są minimum
chi-kwadrat i, w skanie kosmologicznym, profil różnicy chi-kwadrat względem
referencyjnego LCDM liczony przy identycznych ustawieniach.

### 10.3. Seedy i minima lokalne

W JSON-ie sprawdź `multistart_chi2` oraz `search_candidates`. Nie wszystkie
starty muszą trafić do tej samej doliny. Stabilny jest najlepszy wynik, który:

- pojawia się dla więcej niż jednego startu albo jest odzyskiwany przy
  ponownym uruchomieniu z innymi seedami;
- nie pogarsza się po zwiększeniu `de_maxiter` i `de_popsize`;
- jest odzyskiwany przez kontynuację cutoffu;
- nie jest ograniczony przez granicę numeryczną.

### 10.4. Granice

`near_bound=true` oznacza odległość mniejszą niż 1% szerokości boxu. Nie jest
to zarzut wobec fizyczności nuisance parameter, lecz ostrzeżenie, że minimum
może leżeć poza przeszukanym obszarem. Najpierw użyj
`--expanded-counterterm`. Jeżeli parametr nadal leży przy `+/-1000`, trzeba
jawnie zwiększyć box lub zmienić parametryzację i ponowić test stabilności.

### 10.5. Production kontra precision

Wykonaj ten sam fit dla obu bundle'i. Dla skanu modelu ustal z góry tolerancję
na zmianę chi-kwadrat. Praktycznym kryterium startowym jest różnica mniejsza
niż około 0.5, ale ostateczny próg powinien być mniejszy niż efekt fizyczny,
który ma zostać zinterpretowany.

## 11. Uniwersalny notebook

`notebooks/04_analyze_any_model.ipynb` jest przeznaczony dla każdego bundle'a.
W pierwszej komórce konfiguracji zmień tylko:

```python
THEORY_PATH = ...
FIT_PATH = ...
RUN_NEW_FITS = False albo True
K_UV_CUT = 20.0
MODEL_LABEL = ...
```

Notebook kontroluje digest, wypisuje metadane i parametry, rozkłada
chi-kwadrat na redshifty oraz generuje panele danych, pulle i surowe kanały
SPT. Nie zawiera na sztywno liczb z żadnej publikacji.

## 12. Zalecana kolejność dla DCDM lub accDM

1. Odtwórz już zweryfikowany LCDM w tym samym wrapperze CLASS.
2. Zdefiniuj limit nowego modelu, który powinien numerycznie wrócić do LCDM.
3. Wykonaj smoke, potem production i precision dla tego limitu.
4. Porównaj kanały i chi-kwadrat z presetem LCDM.
5. Wykonaj jeden charakterystyczny punkt nowego modelu.
6. Sprawdź production/precision, kilka seedów i cutoffy 10, 15, 20.
7. Dopiero wtedy uruchom siatkę parametrów.
8. Dla każdego punktu zapisuj unikalny `name`, JSON modelu, bundle, fit JSON i
   cutoff scan. Nie nadpisuj wyników poprzedniego punktu.

W DCDM z arXiv:2210.06117 naturalnym pierwszym punktem po limicie LCDM jest
okolica czasu życia około 40 Gyr i `epsilon` około 0.006. Dokładne wartości
wejściowe CLASS muszą jednak odpowiadać konwencji używanej przez konkretną
gałąź `class_decays`.

## 13. Typowe błędy

### `ImportError: No module named lyalpha_pt`

Uruchom z katalogu projektu:

```bash
python -m pip install -e .
```

albo tymczasowo ustaw `PYTHONPATH` na katalog projektu.

### Brak `pk_cb_lin`

Wybrany wrapper CLASS nie udostępnia widma cb. Użyj innej gałęzi/wrappera lub
fizycznie uzasadnij `loop_source=total`. Nie zastępuj tego po cichu.

### CLASS zgłasza nieznany parametr

JSON zawiera nazwę MontePython albo parametr z innej gałęzi CLASS. Najpierw
sprawdź minimalne bezpośrednie wywołanie `Class.set(class_params)`.

### Fit bezpośrednio przy 20 daje 220--230

Sprawdź, czy nie użyto `--no-cutoff-continuation`, starego kodu albo pełnego
sześciowymiarowego fitu. Fiducjalna procedura powinna zawierać ścieżkę
`10 -> 15 -> 20` i profilowanie amplitud.

### Wynik zależy od dziwnych wartości nuisance parameters

Sama duża wartość nie oznacza błędu. Problemem jest dopiero ograniczenie przez
box, niestabilność minimum lub zmiana wyniku kosmologicznego po zmianie
cutoffu/numerics.

## 14. Minimalna checklista przed interpretacją nowego punktu

- [ ] JSON jest bezpośrednio akceptowany przez właściwy CLASS.
- [ ] W modelu nie pozostawiono dwóch alternatywnych gęstości ciemnej materii.
- [ ] `loop_source` i `loop_weight` są fizycznie uzasadnione.
- [ ] Smoke zakończył się i bundle przechodzi `inspect_results.py`.
- [ ] Production i precision zgadzają się na `k<=2 h/Mpc` w ustalonej tolerancji.
- [ ] Fit był wielostartowy, profilowany i prowadzony ścieżką cutoffów.
- [ ] Żadne istotne minimum nie jest ograniczone przez box.
- [ ] Wynik jest sprawdzony dla cutoffów 10, 15 i 20.
- [ ] Fit production i precision daje zgodny chi-kwadrat.
- [ ] Limit modelu odtwarza LCDM przed rozpoczęciem właściwego skanu.

Jeżeli wszystkie punkty są spełnione, ten sam generator, fitter i notebook
mogą być używane bez zmian dla kolejnych punktów modelu.
