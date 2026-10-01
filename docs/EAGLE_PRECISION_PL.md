# Test piku i doliny: CLASS accDM_refactor + one-loop

Dodatek wymaga zastosowanej poprawki Eagle v2. Nie zmienia domyślnego modelu,
starych kampanii ani ich wyników. Wszystkie porównania są nowymi obliczeniami
przy tych samych współrzędnych i założeniach fizycznych Ly-alpha.

## Punkty startowe

Plik `campaigns/accdm/grid_specs/refactor_precision_points.json` można edytować
przed przygotowaniem testu. Wybrano punkty diagnostyczne z obszarów wskazanych
na wykresie; dokładne współrzędne pochodzą z historycznego indeksu siatki
osadzonego w notebooku 13. Nie używamy jego starych chi2 ani wartości uzupełnianych
na mapie. To propozycja do zbadania, nie stwierdzenie przyczyny widocznych skoków.

| Etykieta | log10(m/GeV) | log10(f_acc) |
| --- | ---: | ---: |
| peak_left | 12.4285714285714 | -0.1 |
| peak | 12.7142857142857 | -0.1 |
| peak_right | 13 | -0.1 |
| valley_suspect_1 | 14.7142857142857 | -1.375 |
| valley_suspect_2 | 15 | -1.375 |
| valley_control | 15.7142857142857 | -1.375 |

## Warianty precyzji

| Wariant | background_Nloga | bins/decade | number_tol | SPT |
| --- | ---: | ---: | ---: | --- |
| baseline | 40000 | 50 | 1e-6 | production |
| q100 | 40000 | 100 | 1e-6 | production |
| background80k | 80000 | 50 | 1e-6 | production |
| birthtol1e8 | 40000 | 50 | 1e-8 | production |
| joint | 80000 | 100 | 1e-8 | production |
| joint_spt | 80000 | 100 | 1e-8 | precision |

Zawsze strategia `0, 5`, pełna hierarchia 3, m_nu=0.06,
accdm_smooth_births=0, acc_de_sink=no i automatyczna liczba binów.
Pierwsze trzy zmiany względem baseline dotyczą pojedynczych wejść CLASS.
Zmniejszenie number_tol rozszerza zakres narodzin i może zwiększyć liczbę binów.
Porównanie joint_spt z joint dotyczy dokładności SPT, w tym jego zakresu całkowania:
production ma q_max=50, precision ma q_max=100 h/Mpc.

Każdy przypadek obejmuje CLASS, całki jednopętlowe i oba fity P1D: linear oraz
one_loop. Używamy tych samych pięciu seedów i limitów DE (300 iteracji,
popsize=20). Żaden z wariantów nie jest automatycznie uznawany za zbieżny.

## Uruchomienie na Eagle

Po zainstalowaniu dodatku, z katalogu repozytorium:

```bash
conda activate class_accdm_new
cd /mnt/storage_3/home/asimov/pl0503-01/project_data/lyalpha-one-loop
CLASS_DIR=/mnt/storage_3/home/asimov/pl0503-01/project_data/boltzmannCodes/class_public

python -m campaigns.accdm.precision_suite build \
  --account pl0503-01 --class-source "$CLASS_DIR" \
  --cpus 1 --cpu-budget 40

python -m campaigns.accdm.precision_suite submit --dry-run
python -m campaigns.accdm.precision_suite submit
```

CLASS_DIR musi wskazywać rzeczywiste źródła zainstalowanego wrappera. Jeśli
repozytorium CLASS ma inną nazwę, zmień tę jedną zmienną. Samo build i dry-run
nie zgłasza jobów. Dopiero ostatnie polecenie wysyła tablicę SLURM.

Domyślnie powstaje 6 x 6 = 36 zadań, po 1 CPU, 8 GB i 24 h na zadanie,
w partycji standard. Limit tablicy to floor(cpu-budget/cpus): dla tych ustawień
40, lecz istnieje tylko 36 przypadków. Limit dotyczy tej tablicy, nie innych
Twoich jobów ani gwarantowanego przydziału. SLURM może uruchomić mniej zadań.

W każdym zadaniu teoria i fit wykonują się kolejno, więc nie trzeba utrzymywać
kontrolera ani tmux. Dodatkowe CPU przyspieszają równoległe części CLASS, natomiast
obecne całki SPT i pojedynczy fit działają szeregowo. Np. --cpus 4 --cpu-budget 40
daje maksymalnie 10 jednoczesnych zadań. --cpus 40 daje jedno zadanie naraz;
nie oznacza 40-krotnego przyspieszenia całego pipeline'u.

Ustawienia zasobów, precyzji, punktów i wrappera zapisujemy w suite.json.
Po zmianie ustawień wybierz nowy katalog, np.:

```bash
python -m campaigns.accdm.precision_suite \
  --suite runs/accdm/refactor_precision_4cpu_v1 build \
  --account pl0503-01 --class-source "$CLASS_DIR" --cpus 4 --cpu-budget 40
```

Opcję --suite podaje się przed podkomendą również przy submit/status/report.
--variants baseline q100 pozwala przygotować mniejszy pierwszy test w oddzielnym
katalogu. Nie używaj --software-test-quality smoke do wnioskowania o precyzji;
ta opcja służy wyłącznie szybkim testom działania programu.

## Wyniki, wznowienie i wykresy

```bash
squeue -u "$USER"
python -m campaigns.accdm.precision_suite status
python -m campaigns.accdm.precision_suite report
```

Domyślny katalog:
`/mnt/storage_3/home/asimov/pl0503-01/project_data/lyalpha-one-loop/runs/accdm/refactor_precision_v1`.
Nazwy fizycznych ścieżek /mnt/storage_6 w logach mogą wynikać z rozwiązania
dowiązania /mnt/storage_3; można sprawdzić obie lokalizacje poleceniem readlink -f.

- `suite.json`: punkty, warianty, zasoby, instalacja CLASS i mapowanie task ID.
- `slurm/ARRAY_TASK.out` i `.err`: logi poszczególnych zadań.
- `baseline/`, `q100/` itd.: osobne modele, manifesty, theory/, fits/ i status/.
- `timings/`: czas i kod wyjścia etapów; wznowienie zapisuje czas ostatniej próby.
- `analysis/precision_summary.csv`: chi2, różnica względem baseline dla TEGO SAMEGO
  punktu, nuisance, odległości od granic oraz różnice widm i P1D.
- `analysis/chi2_precision.png`: bezwzględne chi2 i ich zmiany z precyzją.
- `analysis/*_precision.png`: P(k) w całym wspólnym zakresie wejścia SPT,
  kanały one-loop do k_trust oraz P1D przy stałych i ponownie dopasowanych nuisance.

Raport można uruchamiać w trakcie obliczeń. Brakujące i nieprawidłowe wyniki
pozostają oznaczone; nie wstawiamy zer ani wyników interpolowanych. Metadane
modelu, precyzji SPT, instalacji CLASS i digest fitu są sprawdzane przed porównaniem.
Przenoszenie stałych nuisance zachowuje fizyczną amplitudę countertermu mimo
zależnej od teorii normalizacji alpha_ct. Wartości chi2 są ponownie sprawdzane.

Po zakończeniu tablicy i usunięciu przyczyny ewentualnych błędów ponów submit.
Zgłosi tylko przypadki bez kompletu wyników. Worker ponownie sprawdzi istniejącą
teorię, zanim ją wykorzysta. Jeśli tablica jest jeszcze aktywna, submit odmawia
ponownego zgłoszenia. Nie uruchamiaj równolegle zwykłego campaign_manager watch
na podkatalogach tej samej serii testów.

Zmiana samego chi2 może wynikać z optymalizacji nuisance. Dlatego porównujemy
również widma przed fitem, P1D przy stałych nuisance oraz wszystkie znalezione
minima z jednakowych seedów. Surowe wyniki pozostają zachowane.

## Co zweryfikowano lokalnie

Siedem testów automatycznych: izolacja zmian precyzji, 36 przypadków, budżet CPU,
dry-run bez sbatch, niezmienność przygotowanej serii, przekazywanie ścieżek,
niezmienność indeksów tablicy, identyfikator tablicy, blokada duplikatu aktywnej tablicy i blokada fitu po błędzie
teorii. Sprawdzono składnię Bash i instalację poprawki.

Test techniczny użył zweryfikowanej wcześniej teorii ze smoke CLASS/refactor
dla (15,-4), odtworzył marker teorii, wykonał nowy fit z pięcioma seedami i
wygenerował raport. Nie przeliczano lokalnie 36 docelowych przypadków ani nie
zgłaszano zadań na Eagle. Wyniki tych testów kosmologicznych uzyskasz po submit.
