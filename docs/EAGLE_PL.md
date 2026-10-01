# Uruchamianie Ly-alpha na Eagle (SLURM)

Zmiana przygotowana dla kodu Ly-alpha z commita
`d6be43537fa689c288e98fb2cf31b191fd25ca35` oraz interfejsu CLASS
`accDM_refactor` z commita `6d75a45af708e3ec53440eb22cd3a4488960bd48`.

Wersja 2 poprawki używa nowej siatki narodzin CLASS i podanych ustawień
precyzji także w teście technicznym. Profile q1001/q65 z wersji 1 pozostają
dostępne wyłącznie jako jawnie wybierane porównania.

## 1. Środowisko i ścieżki

Docelowy katalog bazowy to
`/mnt/storage_3/home/asimov/pl0503-01/project_data`.
Repozytorium Ly-alpha znajduje się w jego podkatalogu `lyalpha-one-loop`:

```bash
conda activate class_accdm_new
cd /mnt/storage_3/home/asimov/pl0503-01/project_data/lyalpha-one-loop
python -m pip install -e .
```

CLASS musi być zainstalowany z Twojego repozytorium, a nie z PyPI.
Jeżeli wrapper jest już zainstalowany z tego commita `accDM_refactor`, nie
trzeba go ponownie kompilować. Sama wersja `classy` z `pip list` tego nie
potwierdza; standardowy pakiet z PyPI nie implementuje wejść accDM.
Przy instalacji ze źródeł, na przydzielonym węźle obliczeniowym, z katalogu CLASS:

```bash
make -j"${SLURM_CPUS_PER_TASK:-1}" libclass.a
CC=gcc CXX=g++ python -m pip install --force-reinstall --no-build-isolation --no-deps ./python
```

Zachowaj katalog źródeł CLASS w dostępnej na węzłach lokalizacji. Ta metoda
kompiluje ścieżkę do jego tablic zewnętrznych w bibliotece.

W kolejnych przykładach zakładamy katalog CLASS `boltzmannCodes/class_public`.
Jeżeli nadałeś mu inną nazwę, popraw tylko `CLASS_DIR`. `pl0503-01` jest
identyfikatorem projektu z podanej ścieżki; przy innym koncie rozliczeniowym
użyj odpowiedniej wartości `--account`.

```bash
CLASS_DIR=/mnt/storage_3/home/asimov/pl0503-01/project_data/boltzmannCodes/class_public
git -C "$CLASS_DIR" branch --show-current
git -C "$CLASS_DIR" rev-parse HEAD
python -c 'import sys, classy; print(sys.executable); print(classy.__file__)'
```

## 2. Jeden test techniczny

Z katalogu repozytorium Ly-alpha:

```bash
python -m cluster.prepare_eagle \
  --account pl0503-01 \
  --class-source "$CLASS_DIR" \
  --run-dir runs/accdm/eagle_refactor_smoke_v2
```

Polecenie przygotowuje manifest, model i konfigurację, bez zgłaszania jobów.
Domyślnie wybiera jeden punkt `log10(m/GeV)=15`, `log10(f_acc)=-4`:

| Ustawienie | Teoria | Fit |
| --- | --- | --- |
| Partycja | `fast` | `fast` |
| Procesy / rdzenie | 1 / 1 | 1 / 1 |
| Pamięć na job | 8 GB | 4 GB |
| Limit czasu | 1 h | 1 h |
| Dokładność | `smoke` | oba tryby, jeden seed, skrócona optymalizacja |

Test używa profilu `base_model_refactor_birth.json`: pełnej hierarchii,
siatki narodzin 5, `background_Nloga=40000`, `accdm_q_number_tol=1e-6` i
`accdm_q_bins_per_decade=50.0`. CLASS sam wyznacza liczbę binów.
`--quality smoke` obniża wyłącznie rozdzielczość całek SPT i domyślne limity
optymalizacji fitu; nie zmienia parametrów CLASS. Wydruk przygotowania
kampanii pokazuje wybrany profil i wszystkie ustawienia siatki.

`smoke` kontroluje działanie programu i zapis plików. Jego chi2 oraz krzywe
nie są wynikiem naukowym. Nie służy do wyboru rozdzielczości produkcyjnej.

Sprawdź wygenerowaną komendę oraz widoczną kolejkę:

```bash
python -m cluster.campaign_manager submit-theory \
  --campaign runs/accdm/eagle_refactor_smoke_v2 \
  --dry-run --offline --max-points 1

python -m cluster.campaign_manager status \
  --campaign runs/accdm/eagle_refactor_smoke_v2
```

Uruchomienie teorii, a po niej fitu (tylko dla tego jednego punktu):

```bash
python -u -m cluster.campaign_manager watch \
  --campaign runs/accdm/eagle_refactor_smoke_v2 \
  --stages theory fit --max-active 1 --exit-when-complete
```

Kontroler musi pozostać uruchomiony, żeby zakolejkować kolejny etap. Można
użyć dozwolonej przez klaster sesji `tmux`. Przerwanie kontrolera nie anuluje
już zgłoszonego joba. Nie uruchamiaj dwóch kontrolerów tego samego katalogu
kampanii jednocześnie.

Monitorowanie:

```bash
squeue -u "$USER"
python -m cluster.campaign_manager status --campaign runs/accdm/eagle_refactor_smoke_v2
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,AllocCPUS,MaxRSS
```

Wyniki są w `runs/accdm/eagle_refactor_smoke_v2/theory/` oraz `fits/`, względem
katalogu repozytorium pod podaną wyżej ścieżką.
Logi: `logs/theory/POINT.JOBID.out`, `logs/theory/POINT.JOBID.err`
i analogicznie `logs/fit/`. Każda próba zachowuje osobny log.
Joby i markery stanu zapisują identyfikatory SLURM.
Przy `--exit-when-complete` kontroler kończy się kodem 2, jeśli pozostaną
wyłącznie punkty zakończone błędem lub zablokowane przez błąd teorii.

Jeśli job się nie powiedzie, najpierw sprawdź jego `.err` i `sacct`.
Po usunięciu przyczyny odblokuj ponowną próbę:

```bash
python -m cluster.campaign_manager retry-failed \
  --campaign runs/accdm/eagle_refactor_smoke_v2 --stage both
```

Następnie ponownie uruchom `watch`.

## 3. Profil dla nowego CLASS

Plik `campaigns/accdm/base_models/base_model_refactor_birth.json` jest
domyślny w `prepare_eagle` dla `smoke`, `production` i `precision`:

| Parametr | Wartość / znaczenie |
| --- | --- |
| `kappa_acc`, `a_t_acc` | 12.1, 0.133 |
| `eta_acc` | `1e11 / m_acc_in_GeV` |
| `omega_cdm` | wyznaczane przez dotychczasową konwencję stałej wczesnej gęstości |
| `N_ncdm`, `N_ur` | 2, 2.0308: masywne neutrino i córka accDM plus dwa neutrina bezmasowe |
| `m_nu` | 0.06 eV, degeneracja neutrina 1; bez placeholdera w `m_ncdm` |
| masa córki | CLASS bierze ją z `m_acc_in_GeV`; ostatni slot NCDM |
| `ncdm_fluid_approximation` | 3, pełna hierarchia |
| `ncdm_quadrature_strategy` | `0, 5` |
| `ncdm_N_momentum_bins` / `Number of momentum bins` | oba pominięte |
| `background_Nloga` | 40000 |
| `accdm_q_number_tol` | 1e-6 |
| `accdm_q_bins_per_decade` | 50.0 |
| `accdm_smooth_births` | 0 |
| `acc_de_sink` | `no` |
| `evolver` | 0, RKCK |
| `loop_source`, `loop_weight` | `total`, 1 |

Strategia 5 całkuje po siatce logarytmicznej odpowiadającej skalom narodzin
`a_q` od `a_min` do 1. CLASS wyznacza `a_min` z pomijanej frakcji narodzin
`accdm_q_number_tol`, a liczbę punktów z gęstości na dekadę. Jawna liczba binów
ma pierwszeństwo, dlatego usunięto oba jej aliasy. `50.0` nie oznacza 50 binów
łącznie. `accdm_q_schedule="no"` wyłącza starszy dobór q(f) dla strategii 4;
nie wyłącza automatycznej gęstości strategii 5.

Generator obsługuje brak jawnej liczby binów. W `campaign.json` zapisuje wtedy
`accdm_momentum_bins: null` i `accdm_momentum_sampling: "birth_grid_density"`,
obok strategii i wejść precyzji. Nie podszywa się pod CLASS, wyliczając w Pythonie
rzekomą rzeczywistą liczbę binów. Pełne wejście jest również w metadanych `.npz`.

Nowy CLASS sam obsługuje punkty podziału integracji przy narodzinach córki.
Nie trzeba implementować ich w generatorze Ly-alpha. Kod CLASS wymaga
`ncdm_fluid_approximation=3` dla accDM; zachowujemy tę pełną hierarchię.
`acc_de_sink="no"` jest wyborem wariantu fizycznego, nie tolerancją numeryczną.
Zachowano konwencję neutrin, wczesnej gęstości CDM, kappa, a_t i widma całkowitego
z Ly-alpha; nie kopiowano innych, niepodanych założeń modelu weak lensingu.

To żądany punkt startowy precyzji, nie dowód zbieżności likelihoodu Ly-alpha.
Sprawdzanie zbieżności powinno porównywać P(k) w całym zakresie wejścia do
całek SPT, kanały jednopętlowe, P1D i chi2 po ponownym dopasowaniu nuisance.
Rozdzielczość SPT (`--quality`) i precyzja CLASS (`--base-model`) są niezależne.

Generator zapisuje w `.npz` ścieżkę i SHA-256 załadowanego wrappera oraz
zadeklarowany checkout CLASS: commit, gałąź i informację o zmianach śledzonych
plików. Ścieżka źródeł jest deklaracją użytkownika, a nie dowodem, że dany
plik binarny skompilowano z tego checkoutu. Test liczy faktycznie zainstalowany
wrapper. Zmiana wrappera lub commita po przygotowaniu kampanii jest odrzucana.

## 4. Przejście do obliczeń produkcyjnych

Dla tego samego pojedynczego punktu, z pełniejszą optymalizacją:

```bash
python -m cluster.prepare_eagle \
  --account pl0503-01 --class-source "$CLASS_DIR" \
  --run-dir runs/accdm/eagle_refactor_anchor_v2 \
  --quality production --partition standard --walltime 24:00:00
```

Dla istniejącej siatki podaj jawnie plik siatki i model bazowy:

```bash
python -m cluster.prepare_eagle \
  --account pl0503-01 --class-source "$CLASS_DIR" \
  --run-dir runs/accdm/eagle_refactor_validation_v2 \
  --grid-spec campaigns/accdm/grid_specs/validation_grid.json \
  --base-model campaigns/accdm/base_models/base_model_refactor_birth.json \
  --partition standard --walltime 24:00:00 \
  --cpus 1 --memory 8gb --max-active 4 --max-user-active 100
```

Te limity liczby jobów są ustawieniami kontrolera, nie limitami przyznanymi
przez PCSS. Produkcję rozpocznij po przeglądzie testu na Eagle i ustaleniu
docelowych parametrów nowego modelu. Ustawienie większej liczby rdzeni
przyspiesza tylko równoległe części obliczeń; obecne całki SPT i fit nie
dzielą jednego punktu na tyle procesów. Zacznij od 1 rdzenia na punkt.

Każda zmiana modelu, rozdzielczości lub instalacji CLASS wymaga nowego katalogu
kampanii. Nie kopiuj markerów `.done.json`, rekordów jobów ani checkpointów
starej kampanii jako punktu startowego nowej wersji modelu.

## 5. Zgodność z dotychczasowym kodem

Domyślny scheduler istniejących kampanii pozostaje `pbs`. W nowych kampaniach
ustawiany jest `cluster.scheduler="slurm"`; wspólny manager obsługuje oba.
SLURM uruchamia przez Bash istniejące workery teorii/fitu, więc ich nagłówki
`#PBS` nie są interpretowane przez kolejkę. Do zgłaszania na Eagle korzystaj
z managera i nowych `.slurm`, a nie z historycznych komend `qsub` w notebookach
i skryptach pojedynczych, starszych kampanii.

Dokumentacja PCSS:
- https://help.pcss.plcloud.pl/portal/hpc/Partitions%20queues/
- https://help.pcss.plcloud.pl/portal/hpc/4%20Job%20Management%20and%20Scheduling/

PCSS podaje limit 1 h dla `fast` i 168 h dla `standard`; dostęp konta i
aktualne limity konkretnego joba weryfikuje lokalny SLURM.
