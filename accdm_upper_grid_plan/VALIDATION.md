# Weryfikacja pakietu

Przed przekazaniem planu sprawdzono:

- 1122 unikalne punkty górne i cztery rozłączne kontrole dolne; 2252 unikalne ścieżki wymaganych JSON/NPZ.
- Zgodność tożsamości i zapisanych danych odniesienia 30 wyników oraz 175 prób poprzedniej kampanii z pełnym snapshotem 1600 punktów.
- Odtworzenie kanonicznej χ² na rzeczywistym archiwalnym wejściu LCDM: **193.1966105991484** wobec zapisu **193.19661059914839**.
- Mapowanie obu znanych korzeni projektu na nowy `--project-root`.
- Wykrywanie niezgodnego hash `fit.py`, brakującego pliku DR12, zmienionego digestu faktycznej zawartości NPZ, zmienionych parametrów starego JSON, niespójnego startu i brakującego NPZ.
- Obsługę pełnych manifestów przy niedostępnych lokalnie plikach accDM: powstaje raport braków, bez przerwania skryptu nieobsłużonym wyjątkiem.
- Brak zapisów do źródeł i wejść podczas audytu, blokowanie kolizji raportów z wejściami oraz raportowanie błędnych współdzielonych danych DR12 dla każdego punktu; kontrolę składni skryptu PBS.

**Nie sprawdzono tutaj obecności ani kanonicznej χ² wszystkich produkcyjnych punktów accDM.** Ich JSON/NPZ znajdują się pod ścieżkami klastra. Do tego służy dołączony audyt z `--verify-chi2`, uruchamiany w projekcie przez użytkownika.

Pakiet jest planem kampanii i narzędziem audytu. Nie zawiera nowego równoległego runnera fitów ani nowych wyników 1122 punktów.
