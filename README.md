# Asystent Konstytucji RP - Lokalny System RAG

Projekt w 100% lokalnego i bezpiecznego systemu **RAG (Retrieval-Augmented Generation)**, który odpowiada na pytania dotyczące Konstytucji Rzeczypospolitej Polskiej. Aplikacja opiera się na nowoczesnym stosie technologicznym, nie wysyła żadnych danych do zewnętrznych serwerów i korzysta z polskiego modelu językowego **Bielik**.

## 🌟 Główne funkcjonalności

- **Brak halucynacji (Faithfulness):** System posiada twardy _System Prompt_, który wymusza generowanie odpowiedzi **wyłącznie** na podstawie fragmentów Konstytucji.
- **Asynchroniczne strumieniowanie (Streaming):** Odpowiedzi modelu pojawiają się na ekranie płynnie, litera po literze, z wykorzystaniem najnowszego standardu _Responses API_.
- **Statystyki sesji:** Pasek boczny na żywo zlicza zużyte tokeny (Input/Output).
- **Inteligentne cytowanie źródeł:** System podaje dokładny numer artykułu pod odpowiedzią. Jeśli pytanie wykracza poza zakres Konstytucji (Out-of-Domain), asystent grzecznie odmawia i nie wyświetla pustych źródeł.
- **Zarządzanie temperaturą:** Możliwość sterowania "kreatywnością" modelu bezpośrednio z panelu UI.
- **Zarządzanie ilością pobranych artykułów:** Możliwość sterowania ile artykułów jest przekazywane do LLMa.
- **Buforowanie odpowiedzi (cache):** Aplikacja zapamiętuje odpowiedzi modelu w pamięci operacyjnej na czas trwania sesji. Ponowne zadanie identycznego pytania zwraca wynik natychmiastowo — bez wywołania LLM — zachowując przy tym efekt wizualny strumieniowania. Klucz bufora zależy od trzech parametrów: **treści pytania**, **temperatury** oraz **Top K**, dzięki czemu zmiana ustawień skutkuje wygenerowaniem świeżej odpowiedzi. W logach odnotowywana jest liczba zaoszczędzonych tokenów (input i output) przy każdym trafieniu w bufor.

## 🛠️ Stos technologiczny

- **Język:** Python 3.10+
- **Interfejs Użytkownika:** [Streamlit](https://streamlit.io/)
- **Baza Wektorowa:** [Qdrant](https://qdrant.tech/) (uruchamiana lokalnie w trybie bezserwerowym)
- **Embeddingi:** `fastembed` (model: `intfloat/multilingual-e5-large`)
- **Lokalny LLM:** [LM Studio](https://lmstudio.ai/) + [Bielik 11B](https://huggingface.co/speakleash)
- **Logowanie:** `loguru`
- **Menedżer pakietów:** `uv`

\---

## 🚀 Jak uruchomić projekt?

### Krok 1: Klonowanie repozytorium i instalacja zależności

Do zarządzania pakietami zalecane jest użycie [uv](https://github.com/astral-sh/uv).

```bash
# 1. Sklonuj repozytorium
git clone https://github.com/tomaszwolk/SGGW_Glebokie_sieci_neuronowe_zadanie.git

# 2. Zainstaluj wymagane biblioteki
uv sync
```

### Krok 2: Uruchomienie lokalnego modelu w LM Studio

Aplikacja wykorzystuje LM Studio jako lokalny serwer zgodny z API OpenAI.
Pobierz i zainstaluj LM Studio. (Upewnij się, że masz wersję 0.3.29 lub nowszą, która wspiera Responses API).
W wyszukiwarce LM Studio (ikona lupy) wpisz: Bielik-11B-v2.3-Instruct.
Pobierz wersję w formacie GGUF dopasowaną do Twojej pamięci RAM/VRAM (np. Q4_K_M lub Q5_K_M – wymagają ok. 6-8 GB).
Przejdź do zakładki Local Server (ikona <-> na lewym pasku).
Z górnego menu rozwijanego wybierz pobranego Bielika i poczekaj, aż model się załaduje do pamięci.
Kliknij zielony przycisk Start Server.
Upewnij się, że serwer działa na domyślnym porcie 1234. Zobaczysz komunikat typu: Server listening on http://localhost:1234/v1.

### Krok 3: Uruchomienie aplikacji

Kiedy serwer LM Studio działa, uruchom interfejs:

```bash
cd src
uv run streamlit run app.py
```

Uwaga: Przy pierwszym uruchomieniu skrypt pobierze z sieci model embeddingów (ok. 2.2 GB). Po zakończeniu w folderze głównym pojawi się katalog qdrant_db/.

Aplikacja otworzy się automatycznie w Twojej przeglądarce pod adresem http://localhost:8501.

### Struktura projektu

```
├── data/
│   └── konstytucja_rp_z_dnia_2_04_1997.md # Plik źródłowy dokumentu
├── src/
│   ├── qdrant_db/ # (Generowany) Baza wektorowa
│   ├── app.py # Główny interfejs aplikacji (Streamlit)
│   ├── chunker.py # Logika parsowania Markdowna i podziału na artykuły
│   ├── config_loader.py # Skrypt ładujący konfigurację
│   ├── config.yaml # Zcentralizowana konfiguracja systemu
│   ├── indexer.py # Generowanie wektorów i zapis do Qdrant
│   ├── rag_logic.py # Główny silnik RAG i komunikacja z API LLM
│   ├── stream_utils.py # Parsowanie streamu, buforowanie odpowiedzi
│   └── logs/
│       └── rag_app.log # (Generowany) Plik z logami systemu
└── README.md
```

### Konfiguracja (config.yaml)

Wszystkie kluczowe parametry systemu znajdują się w pliku config.yaml. Możesz tam łatwo zmienić:
Ścieżki do plików.
Model embeddingów oraz liczbę wyszukiwanych fragmentów (top_k).
Prompty systemowe i parametry serwera LM Studio (adres URL, limity tokenów).

### 📝 Przykładowe pytania

- Aby przetestować działanie systemu, spróbuj zadać następujące pytania w czacie:
- Kto może zostać prezydentem Polski?
- W jakich sytuacjach można wprowadzić stan wyjątkowy?
- Kto uchwala budżet państwa?
- Jak upiec sernik? (Test systemu zabezpieczającego przed halucynacjami).
