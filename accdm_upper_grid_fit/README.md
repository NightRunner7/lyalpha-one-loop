# accDM: pełna górna siatka, etap A

Pakiet uruchamia **1122 punkty górnej siatki i cztery kontrole** bez osobnej partii pilotażowej. Korzysta z zapisanych widm, oryginalnych zakresów nuisance, kanonicznej funkcji χ² i analitycznego profilera sprawdzonego w kampanii 30 punktów.

## Uruchomienie na klastrze

Rozpakuj ZIP w głównym katalogu `lyalpha_one_loop`. Powinien powstać katalog `accdm_upper_grid_fit/`. Aktywuj to samo środowisko, w którym przeszedł audyt, a następnie:

```bash
export PROJECT_ROOT="$PWD"
export PYTHON_BIN="$(command -v python)"
qsub -v PROJECT_ROOT,PYTHON_BIN accdm_upper_grid_fit/run_full.pbs
```

To jest job **fitujący**, nie kolejny audyt. Rezerwuje jeden węzeł, **8 CPU, 8 GB RAM, do 8 godzin**. Osiem procesów równolegle dopasowuje różne punkty; każdy ma jeden wątek BLAS. Osiem godzin jest limitem rezerwacji, nie zmierzonym czasem kampanii. Czas całej siatki nie został jeszcze zmierzony.

Wyniki trafiają do:

```text
runs/accdm/reprofile_upper_v2/stage_A/
```

Ponowne wysłanie tej samej komendy wznawia obliczenia: ukończone zadania z niezmienionymi wejściami są odczytywane z dysku. Dwa joby nie mogą jednocześnie pisać do tego samego katalogu. Nie łącz tego stanu z wcześniejszym `reprofile_30_v1`.

Jeżeli lokalne reguły kolejki wymagają innej rezerwacji CPU, zmień **zarówno** `ncpus=8`, jak i `--workers 8` w PBS. Parametry 32 rdzeni z innych symulacji nie są tutaj potrzebne.

Bezpośrednie uruchomienie na przydzielonym węźle obliczeniowym:

```bash
python -u accdm_upper_grid_fit/run_upper.py \
  --project-root "$PWD" --phase full --workers 8 --max-rounds 12 \
  --out runs/accdm/reprofile_upper_v2/stage_A --resume
```

## Co robi kampania

1. Sprawdza bieżące hashe JSON, NPZ, kodu i danych względem zakończonego audytu. Kopie zaakceptowanych raportów są już w `audited_inputs/`; nie musisz ich przenosić ani uruchamiać audytu od nowa.
2. Tworzy bank startów ze starych fitów, 30 poprawionych wyników i konkurencyjnych rozwiązań zapisanych w 175 próbach. Dane i stare fity pozostają niezmienione.
3. W każdej rundzie przekazuje własne rozwiązania, rozwiązania najbliższych zapisanych sąsiadów na osiach i do ośmiu reprezentantów różnych rozwiązań z całej fitowanej siatki. Sąsiedztwo może obejmować lukę siatki: przekazywany jest wyłącznie start, bez interpolowania widm lub χ².
4. Ponownie profiluje amplitudy na teorii celu **przed wyborem startów**. Odrzuca niedopuszczalne parametry kształtu; nie obcina ich do granic. Obce amplitudy mogą leżeć poza domeną, ponieważ są wyznaczane na nowo.
5. Dopasowuje maksymalnie cztery różne starty: dwa najlepsze oraz do dwóch zwiększających różnorodność. Budżet podstawowy to 2000 ocen, a po nieudanej próbie 5000. Następnie wykonuje pełne sześcioparametrowe dopracowanie od maksymalnie dwóch różnych kandydatów.
6. Zachowuje najlepszy znany wynik i bank do sześciu różnych rozwiązań na punkt. Każdy wektor jest sprawdzany kanoniczną funkcją celu. Nieudany solver nie kasuje wcześniej lepszego wyniku.
7. Scala wyniki dopiero po całej rundzie. Nowo znalezione rozwiązania docierają do sąsiadów w następnej rundzie; kolejność zakończenia procesów nie zmienia startów bieżącej rundy.

Maksymalnie 12 rund, z wcześniejszym zakończeniem po dwóch stabilnych rundach. Robocza tolerancja poprawy: 0.01 w χ², a w pasie `|Δχ²−5.991|≤1` — 0.003. Sprawdzana jest również stabilność banków i udany końcowy test lokalny. Osiągnięcie limitu rund bez stabilizacji jest jawnie oznaczone jako nierozstrzygnięte.

## Wyniki do przesłania

Po każdej pełnej rundzie dostępne są:

- `summary.json` — postęp, poprawy, liczba nierozstrzygniętych punktów i status stabilizacji;
- `reprofile_best_points.csv` — najlepsze parametry, χ², poprawa i bliskość granic;
- `latest_round_attempts.csv` — próby ostatniej ukończonej rundy;
- `candidate_bank.json` — zachowane rozwiązania;
- `rounds/NNN/*.json` — pełna historia poszczególnych punktów, w tym starty, próby, walidacja i czas wykonania.

Na początek wystarczą `summary.json` i `reprofile_best_points.csv`. Jeśli job kończy się kodem 2, prześlij komunikat `BLOCKED` z logu PBS oraz `rounds/NNN/failures.json`, jeśli powstał. Po poprawieniu przyczyny wznowienie wykorzysta gotowe punkty tej rundy.

`chi2_old` odnosi się do oryginalnego fitu, a `chi2_best` zawiera też znane już poprawy 30 punktów. Raport rozróżnia Δχ² względem starego minimum i zaktualizowanego minimum całego snapshotu. Dolne punkty poza czterema kontrolami pozostają przy wcześniejszych wynikach.

## Zakres wniosków

To pełna **kampania lokalnego profilowania z propagacją kilku rozwiązań**, przy zachowaniu starych zakresów. Nie zmienia countertermów ±100 na ±1000 i nie rozszerza `beta_bias`. Nie generuje nowej teorii. Kontrole globalne trudnych punktów oraz etapy B/C/D pozostają oddzielnymi działaniami po obejrzeniu wyniku tej kampanii. Stabilizacja rund nie dowodzi znalezienia minimum globalnego.

Audyt potwierdził 1126/1126 wejść i odtworzenie starej χ² z różnicą 0.0. Ustawienia całkowania one-loop są wspólne, lecz rzeczywiste siatki CLASS nie są jednakowe:

| Biny pędu accDM | `background_Nloga` | Punkty górne |
|---:|---:|---:|
| 1001 | 5001 | 443 |
| 2501 | 5001 | 7 |
| 5001 | 5001 | 583 |
| 5001 | 1001 | 89 |

Nazwy kampanii nie zastępują tych metadanych. Górne punkty mają oryginalne zakresy CT: 587 razy ±100 oraz 535 razy ±1000. Kontrole dodają po dwa punkty obu rodzajów.

## Odtwarzalność

Nowy stan zawiera hashe kodu, wejść, audytu i wersje Python/NumPy/SciPy. Zmiana tych elementów blokuje wznowienie starego stanu. Liczbę procesów można zmienić, a limit rund zwiększyć bez unieważniania ukończonych zadań. Nie zmieniaj wówczas budżetów ani zestawu startów.

Kod został sprawdzony lokalnie na archiwalnym widmie LCDM oraz testach transportu/startów/wznowienia. Nowe fity produkcyjne accDM zostaną wykonane dopiero po wysłaniu joba na Twój klaster.
