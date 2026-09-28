# Profilowanie całej górnej siatki accDM

## Zakres i status

To jest konkretny plan nowej kampanii, zamrożony manifest wejść oraz działający audyt dostępności i zgodności danych. Pakiet nie zawiera jeszcze nowego, równoległego runnera profilowania. Dołączony skrypt PBS uruchamia audyt, nie fity. Nie uruchomiono niczego na klastrze.

Za górną siatkę przyjmujemy gęstą część: **−1.6 ≤ log10(f_acc) ≤ 0**. Obejmuje ona zarówno główne oscylacje, jak i dwa już naprawione punkty przy −1.375.

- **1122 zapisane punkty**, 60 wartości log10(m_acc) od 11 do 19.4285714286 oraz 21 poziomów frakcji.
- Cztery dodatkowe kontrole przy log10(f_acc)=−4: log10(m_acc)=11, 14, 17, 19.2857142857.
- Pełny prostokąt 60×21 miałby 1260 komórek. **138 pozycji nie występuje w przekazanym snapshotcie**. Nie klasyfikujemy ich jako failed jobs ani nie uzupełniamy interpolacją. Lista: `grid_gaps.csv`. Przy m≤18 brakuje 25 komórek; pozostałe 113 leżą powyżej 18, głównie poza obszarem gęstego skanu.
- Dolne 478 zapisanych punktów pozostają w pełnym indeksie odniesienia. Wektory nuisance z 60 punktów przy log10(f_acc)=−1.9 są dodatkową pulą startów przy dolnym brzegu; nie trzeba liczyć ich widm ponownie.

Manifest odzwierciedla dane przekazane w tej rozmowie. Nie obejmuje automatycznie późniejszych wyników klastra. Jeśli powstały nowe widma/punkty, trzeba przygotować aktualny spójny snapshot przed zamrożeniem kampanii. Wybór teorii dla jednej współrzędnej powinien wynikać z ustalonej polityki jakości i pochodzenia, nie z najmniejszej χ² uzyskanej na różnych widmach.

## Dane są rozproszone, ale nie trzeba ich scalać

Wszystkie ścieżki górnych punktów w snapshotcie prowadzą do `runs/accdm/` w projekcie. Audyt czyta je bezpośrednio:

| Podkatalog kampanii | Punkty górne |
|---|---:|
| `accdm_scan_highf_q5001_recovery_10core_v1` | 476 |
| `accdm_scan_highf_q5001_v1` | 99 |
| `accdm_scan_lowf_q1001_v2` | 443 |
| `accdm_lowf_recovery_q2501_10core_v1` | 7 |
| `accdm_scan_loweps_lowf_q1001_v1` | 89 |
| `accdm_scan_loweps_highf_q5001_v1` | 8 |

**Jeżeli te katalogi nadal są dostępne na klastrze, nie trzeba przenosić starych wyników fitu.** W czasie poprzedniego eksportu wszystkie 1122 miały zapisane flagi complete/available. Bieżącą obecność plików musi jednak sprawdzić audyt uruchomiony na klastrze.

Potrzebne są:

1. Wskazane w `required_point_files.csv` oryginalne **JSON fitu i odpowiadające NPZ teorii**: 2244 pliki górne oraz 8 plików dla czterech kontroli. Nie wystarczą same końcowe wartości χ².
2. Zgodny katalog `lyalpha_pt/` oraz dotychczasowy pakiet `accdm_reprofile_30_points/` do kanonicznej walidacji. Obsługiwany SHA256 `fit.py`: `845ec6896c8901df8535d2bc8069344842235e1fc38ba4bca7fd5a59acfb30b6`.
3. Trzy pliki DR12: `Pk1D_data.dat`, `Pk1D_syst.dat`, **`Pk1D_cor.dat` również przy paper_diag**. Ich katalog jest odczytywany ze starego JSON; można jawnie podać `--data-dir` po przeniesieniu.
4. Wyniki ostatniej kampanii: `runs/accdm/reprofile_30_v1/reprofile_best_points.csv` i `reprofile_attempts.csv`. Kopie przesłanych przez Ciebie 30 wyników i 175 prób są także dołączone do tego pakietu jako `seed_best_points.csv` i `seed_attempts.csv`.
5. Pełny snapshot 1600 punktów: `baseline_full_snapshot.csv`, dołączony do pakietu.

Nie są wymagane logi generatora, checkpointy CLASS ani ponowne obliczenia one-loop. Stare manifesty i statusy warto zachować jako informację o zasięgu skanu, lecz nowe dopasowanie nie powinno uzależniać od nich odczytu poprawnego JSON/NPZ.

Jeżeli brakuje danych, kopiujemy tylko wskazane pliki, **z zachowaniem względnej struktury katalogów**. Nie spłaszczamy ich do jednego `fits/` lub `theory/`, bo różne kampanie mogą zawierać podobne nazwy. Oba znane korzenie projektu — klastrowy `/home/2/ks405818/Master/lyalpha_one_loop` i lokalny `/home/krzysztof/Workspace/lyalpha_one_loop` — są mapowane na podany `--project-root`. Inne ścieżki bezwzględne są raportowane jawnie. Nie modyfikujemy digestów ani wnętrza NPZ.

**Nie zastępuj parametrów w starym snapshotcie nowymi wynikami, pozostawiając stare JSON.** To zniszczyłoby kontrolę zgodności wejść. Nowe wektory importujemy do osobnego banku startów i ponownie oceniamy na teorii punktu docelowego. Nowa kampania otrzyma własny stan w `runs/accdm/reprofile_upper_v2/`; nie używamy `--resume` starego stanu 30 punktów do nowego manifestu.

## Ważne odkrycie: dwa różne zakresy countertermów

W górnej siatce **587 punktów** ma zapisane ograniczenia `alpha_ct, beta_ct ∈ [−100,100]`, a **535 punktów** ma `[−1000,1000]`. Pozostałe granice są wspólne. Wszystkie mają `paper_diag` i k_UV=20 h/Mpc.

Dlatego kampanię dzielimy na osobne etapy:

| Etap | Teoria | Zakres nuisance | Cel |
|---|---|---|---|
| A: poprawa profilowania | te same NPZ | dokładnie stare granice każdego punktu | oddzielić błąd optymalizacji od zmiany domeny |
| B: wspólna domena | te same NPZ | oba CT w [−1000,1000], beta_bias nadal [−15,20] | uzyskać porównywalne profile górnej siatki |
| C: czułość na granicę | te same NPZ | osobne scenariusze beta_bias_min=−20 i −30 | sprawdzić wpływ technicznej dolnej granicy |
| D: dokładność teorii | nowe wybrane NPZ | ustalona domena z wcześniejszego etapu | sprawdzić struktury, które przetrwały stabilne profilowanie |

Etap B to proponowana wspólna domena numeryczna, nie nowy fizyczny prior. Różnice A→B raportujemy oddzielnie od poprawy starego fitu→A. Jeśli wspólna domena ma być użyta do końcowych ograniczeń całej mapy, również dolna część i minimum odniesienia muszą zostać doprowadzone do tej samej konwencji; nie łączymy bez oznaczenia scenariuszy o różnych domenach.

Etap C zaczynamy od punktów przy granicy oraz konturze i ich sąsiadów. Zapisane granice same w sobie nie dowodzą, że szerszy zakres jest fizycznie dopuszczalny. Jeśli test wykazuje istotny wpływ, wynik wykluczenia pozostaje zależny od wyboru domeny; przed końcową mapą przyjmujemy jedną uzasadnioną konwencję.

## Jak ma działać nowy profiler

### A0. Weryfikacja i import

Odtworzyć starą χ², sześć parametrów, bounds, digest teorii, dane i kowariancję. Zachować hash kodu i rzeczywiste metadane precyzji NPZ. Nazwy kampanii `q1001/q5001` traktujemy jako pochodzenie; nie jako samodzielne potwierdzenie ustawień CLASS. Zgodnie z CSV górna siatka zawiera 532 punkty oznaczone 1001, 7 oznaczonych 2501 i 583 oznaczone 5001 — ewentualnych różnic teorii profilowanie samo nie usuwa.

Zaimportować wyniki 30 punktów: 29 należy do nowej górnej siatki, jeden jest dolną kontrolą. Ich stare χ² i digests zgadzają się z pełnym snapshotem. Wszystkie nowe wektory trzeba jednak ponownie ocenić przed użyciem. Przechowywać także konkurencyjne rozwiązania z tabeli prób, nie tylko zwycięzcę.

### A1. Różnorodne starty i bank kilku rozwiązań

Każdy punkt dostaje stary fit, dostępny nowy fit, starty sąsiednie oraz początkowo 6–10 reprezentantów różnych rozwiązań z kampanii 30 punktów. Dopuszczalność kształtu sprawdzamy w domenie punktu docelowego.

**Starty oceniamy po ponownym, ograniczonym profilowaniu dwóch amplitud na teorii celu. Nie sortujemy ich według surowej χ² przeniesionego sześciowektora.** W poprzedniej kampanii zwycięskie starty miały przed profilowaniem χ² rzędu 6464 i 11343, po profilowaniu około 220 i 226, a po optymalizacji około 215. Ranking surowej χ² mógłby odrzucić właśnie te rozwiązania.

Aktywny bank: do sześciu różnych rozwiązań na punkt. Utrzymywać różnorodność m.in. znaków `alpha_bias` i zakresów `beta_ct`. Są to cechy pomagające szukać różnych rozwiązań, nie samodzielna definicja gałęzi. Deduplikacja uwzględnia cztery parametry kształtu i różnice przewidywań P1D w jednostkach kowariancji. Bliska χ² nie oznacza identycznego rozwiązania. Nie stosować standardowej tolerancji absolutnej do `alpha_ct≈10⁻⁸⁷`; zachować pełne wektory w historii.

### A2. Rundy propagacji po całej siatce

Na początku rundy zamrozić bank startów. Wszystkie joby korzystają z tej samej wersji. Dla każdego punktu zebrać własne gałęzie, gałęzie najbliższych zapisanych sąsiadów w czterech kierunkach osi i wybranych nowych reprezentantów z innych części mapy. Długość krawędzi i ewentualna luka siatki są jawne. Donor dostarcza tylko start; niczego nie interpolujemy w teorii ani w χ².

Profilować amplitudy dla wszystkich dopuszczalnych propozycji, następnie optymalizować dwa najlepsze oraz do dwóch dodatkowych różnych startów. Domyślny budżet lokalny: **2000 ocen**, eskalacja do **5000** dla przypadków nierozstrzygniętych. Stary lub lepszy odwiedzony wektor pozostaje zachowany także po niepowodzeniu solvera, ale nie dostaje z tego powodu statusu zbieżności.

Po zakończeniu całej rundy scalić wyniki i dopiero wtedy przekazać je następnej rundzie. Kolejność wykonania jobów nie zmienia wejść w obrębie rundy. Taka runda nie jest równoważna sekwencyjnemu przejściu: nowa gałąź przechodzi najwyżej jedną krawędź na rundę. Dlatego początkowy wspólny bank i okresowe przekazywanie reprezentantów do całej mapy są ważne.

Początkowy limit organizacyjny: 12 rund. Dwie kolejne rundy powinny nie dawać istotnej poprawy ani nowych zachowanych gałęzi. Robocze progi: **0.01 w χ²**, **0.003 dla punktów w pasie |Δχ²−5.991|≤1**; błąd numerycznej oceny musi być znacznie mniejszy. W starym snapshotcie pas ten obejmuje 157 górnych punktów; po zmianie minimum pas trzeba przeliczyć. Osiągnięcie limitu rund z aktywną propagacją oznacza wynik nierozstrzygnięty.

### A3. Poszukiwanie pominiętych rozwiązań

Nie polegać wyłącznie na jednolitym DE w ogromnym zakresie beta_ct. Dla trudnych kotwic i nierozstrzygniętych punktów użyć osobnych obszarów:

`[-1000,-100], [-100,-20], [-20,0], [0,20], [20,100], [100,1000]`, przeciętych z aktualną domeną.

Uwzględnić zero i granice; lokalne dopracowanie może następnie poruszać się w całej domenie. Budżet na trudną kotwicę: początkowo **20–40 tys. ocen**, rozdzielony na obszary i dwa ziarna. Kotwice obejmują dotychczasowe trzy punkty, minimum odniesienia, nierozstrzygnięty `(14.7142857143, −0.4)`, wybrane największe lokalne odchylenia i punkty przy granicach. Nie wykonujemy tak kosztownego szukania automatycznie dla wszystkich 1122 punktów.

Nowe rozwiązania znalezione w tym etapie trafiają do kolejnych rund propagacji. Sama zgodność dwóch DE ani komunikat sukcesu nie zamyka testu: w poprzednich danych oba DE przeoczyły udowodnione lepsze rozwiązania w dwóch kotwicach.

### A4. Kontrola końcowa i raport

Pełne sześcioparametrowe dopasowanie od najlepszego rozwiązania dla każdego punktu oraz od konkurencyjnych rozwiązań w miejscach rozbieżności. Dodatkowo wybrane przekroje w obu kierunkach. Porównywać kanoniczne końcowe χ² solverów, a nie tylko zachowany najlepszy kandydat. Full6 pełni tu rolę kontroli lokalnej; dotychczasowe duże poprawy odkrywała kontynuacja profilowania czterech parametrów.

Raportować mapy absolutnej χ² i Δχ², poprawę, status i liczbę gałęzi, najbliższą granicę, źródło NPZ, P1D i rozbicie po redshiftach oraz pokrycie siatki. Wyznaczać kontury dopiero po kontroli ich stabilności; nie wygładzać błędnych fitów interpolacją.

## Odniesienie χ² i końcowe ograniczenia

Dotychczasowe minimum pełnego snapshotu: **χ²=188.42959746601295** przy `(12.7142857143, −0.2)`. Ten punkt jest w nowej kampanii; nie był w pilocie 30 punktów.

Do porównania przed/po używać wspólnego starego odniesienia. Po zakończeniu etapu A policzyć minimum z najlepszych dostępnych wyników całej siatki: nowych górnych i kontrolnych oraz pozostałych starych dolnych. Jawnie oznaczyć, że reszta dolnej siatki nie była ponownie profilowana. Po zmianie wspólnej domeny w B/C odniesienie musi pochodzić z tego samego scenariusza; nie mieszać go z A. Zachować absolutne χ², żeby móc przeliczyć wszystkie różnice spójnie.

## Organizacja na klastrze

Nowy runner będzie potrzebny: dotychczasowy `run_reprofile.py` ma na sztywno przekroje i kontrole 30 punktów. **Samo podanie mu manifestu 1122 punktów nie wystarczy.** Nie współdzielić jego jednego `state.json` pomiędzy jobami.

Proponowany podział nowej kampanii: około **32 punktów na job, 1 CPU i jeden wątek BLAS**, na początek do **8 równoległych jobów**. Każdy job ma własny plik wynikowy i punkt kontrolny; scalanie odbywa się po całej rundzie. Wersja banku, scenariusz granic, hash wejść i kodu należą do tożsamości joba. Wznowienie powtarza wyłącznie nieukończone próby.

Najpierw partia pomiarowa około 32 reprezentatywnych punktów powinna zmierzyć pełny czas wczytywania, walidacji i fitów oraz pamięć. Z tego dobieramy walltime partii. Poprzednie 175 prób trwało łącznie około 36 s samej optymalizacji, lecz nie obejmuje to I/O, kontekstów, eksportu ani kosztu rozszerzonej strategii. Większa liczba CPU nie przyspiesza automatycznie pojedynczego fitu.

## Co uruchomić teraz

Rozpakuj paczkę do głównego katalogu projektu. Powstanie `accdm_upper_grid_plan/`. Stare katalogi wynikowe pozostają na miejscu.

Bezpośrednio, w środowisku używanym do pilota, na węźle obliczeniowym:

```bash
python accdm_upper_grid_plan/audit_upper_inputs.py \
  --project-root "$PWD" \
  --seed-dir runs/accdm/reprofile_30_v1 \
  --out runs/accdm/reprofile_upper_v2/preflight \
  --verify-chi2
```

Albo jako job PBS, z głównego katalogu projektu i aktywnego środowiska Pythona:

```bash
export PROJECT_ROOT="$PWD"
export PYTHON_BIN="$(command -v python)"
qsub -v PROJECT_ROOT,PYTHON_BIN accdm_upper_grid_plan/preflight_upper.pbs
```

Skrypt PBS jest wzorowany na składni `select=1:ncpus=1:mem=4gb` używanej w projekcie; rezerwuje do godziny na audyt. Nie generuje widm ani nie uruchamia optymalizacji. Domyślnie weryfikuje też χ² przez istniejący kanoniczny fitter.

Audyt zapisze `preflight_summary.json`, `preflight_points.csv` i `missing_files.csv`. Bez opcji `--verify-chi2` sprawdza pliki i metadane, ale nie potwierdza odtworzenia funkcji celu. Jeśli jakiś plik zniknął lub zmienił się względem snapshotu, raport poda konkretną ścieżkę i przyczynę.

**Do dalszego przygotowania uruchomienia potrzebne są raport podsumowania i lista braków, a przy błędach zgodności także `all_issues.csv`.** Dzięki nim można dokładnie ustalić transfery, zaktualizować manifest w razie nowych wyników i skonfigurować nowy runner bez zgadywania ścieżek. Zakres lokalnych sprawdzeń pakietu opisuje `VALIDATION.md`; nie zastępują one audytu produkcyjnych plików na klastrze.
