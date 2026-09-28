# Zakres sprawdzeń

Wejścia produkcyjne zostały sprawdzone przez przesłany audyt klastra: 1126/1126 punktów, komplet JSON/NPZ, zero błędów i różnica odtworzonej starej χ² równa 0.0. Nowy runner używa kopii tych raportów i kontroluje hashe bieżących plików.

Nowy worker sprawdzono na rzeczywistym archiwalnym widmie LCDM. Odtwarza χ²=193.1966105991484; najlepszy uzyskany wynik różni się od niego jedynie na poziomie zaokrągleń. Są to testy kodu z syntetycznymi etykietami współrzędnych, nie nowe wyniki naukowe accDM.

Sprawdzone przypadki:

- dokładne powiązanie zapisywanej χ² z zapisywanym sześciowektorem;
- profilowanie donorów o amplitudach poza granicami celu, gdy cztery parametry kształtu są dopuszczalne;
- odrzucanie niedopuszczalnego kształtu bez obcinania do granic;
- zachowanie poprzedniego najlepszego wyniku;
- blokowanie niezgodnych hashy i nieodtwarzalnych wartości χ² banku;
- brak zmian w oryginalnym kodzie i danych;
- deterministyczne sąsiedztwo i propozycje, bez modyfikowania banku bieżącej rundy;
- odczyt rzeczywistych 1600 starych wyników, 30 najlepszych nowych wektorów i 175 prób;
- składnia PBS i interfejs CLI.

Test integracyjny równoległości i wznowienia jest opisany w dołączonym `integration_qa.json`. Nie uruchomiono nowej kampanii na Twoim klastrze. Status lokalnej stabilizacji nie jest certyfikatem minimum globalnego.
