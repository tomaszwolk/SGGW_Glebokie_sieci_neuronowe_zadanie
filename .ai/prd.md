# Product Requirements Document (PRD)
**Nazwa projektu:** Konstytucja RP - Lokalny System RAG
**Wersja:** 1.0
**Data:** Kwiecień 2026

## 1. Cel Projektu
Stworzenie lokalnego, bezpiecznego systemu Q&A (Pytania i Odpowiedzi) opartego na architekturze RAG (Retrieval-Augmented Generation), który pozwala użytkownikowi zadawać pytania w języku naturalnym dotyczące Konstytucji Rzeczypospolitej Polskiej. System ma udzielać precyzyjnych odpowiedzi opartych **wyłącznie** na tekście Konstytucji, wskazując konkretne źródło (numer artykułu/rozdziału).

## 2. Architektura i Stos Technologiczny
Projekt opiera się na nowoczesnych, otwartoźródłowych technologiach i architekturze "Vanilla RAG" (bez ciężkich frameworków typu LangChain).

*   **Dane źródłowe:** Konstytucja RP w formacie Markdown (`.md`).
*   **Podział tekstu (Chunking):** Semantyczny – podział na pojedyncze Artykuły wraz z metadanymi (np. nazwa Rozdziału).
*   **Model Embeddingów:** `intfloat/multilingual-e5-large` (standard dla wielojęzycznych wektorów, świetnie radzi sobie z j. polskim).
*   **Baza Wektorowa:** Qdrant (uruchamiany lokalnie).
*   **Model LLM:** Bielik (np. wersja 11B Instruct) uruchomiony lokalnie za pomocą LM Studio (kompatybilność z OpenAI API). Wymagania sprzętowe: < 16GB VRAM.
*   **Logika RAG:** Autorski kod w Pythonie (Vanilla RAG), wydzielony do osobnego modułu.
*   **Interfejs Użytkownika (UI):** Streamlit (okno czatu w przeglądarce).
*   **Logowanie:** Biblioteka `loguru` (zapis do pliku).

## 3. Wymagania Funkcjonalne (MVP)
1.  **Przetwarzanie i Indeksowanie (Data Ingestion):**
    *   Skrypt wczytujący plik `.md`.
    *   Ekstrakcja artykułów i przypisywanie im metadanych (Rozdział).
    *   Generowanie embeddingów dla każdego artykułu.
    *   Zapis wektorów i metadanych do bazy Qdrant.
2.  **Wyszukiwanie (Retrieval):**
    *   Przyjęcie zapytania od użytkownika.
    *   Zamiana zapytania na wektor (embedding).
    *   Wyszukanie *top-k* (np. 3-5) najbardziej zbliżonych semantycznie artykułów w bazie Qdrant.
3.  **Generowanie Odpowiedzi (Generation):**
    *   Zbudowanie promptu zawierającego: instrukcję systemową, wyszukane fragmenty Konstytucji (kontekst) oraz pytanie użytkownika.
    *   Wysłanie zapytania do lokalnego modelu Bielik przez API LM Studio.
    *   Zwrócenie odpowiedzi do użytkownika wraz z podaniem źródła (np. "Zgodnie z Art. 127...").
4.  **Interfejs Czatowy:**
    *   Aplikacja Streamlit z historią konwersacji.
    *   Pole tekstowe do wprowadzania zapytań.
    *   Wyświetlanie odpowiedzi asystenta.
5.  **Logowanie (Loguru):**
    *   Zapisywanie do pliku logów (np. `rag_app.log`) informacji o:
        *   Otrzymanym zapytaniu.
        *   Czasie wyszukiwania i wygenerowania odpowiedzi.
        *   Wyszukanych fragmentach (kontekście).
        *   Wszelkich błędach (np. brak połączenia z LM Studio, błędy bazy Qdrant).

## 4. Wymagania Niefunkcjonalne
*   **Brak Halucynacji (Faithfulness):** Model ma bezwzględny zakaz używania wiedzy spoza dostarczonego kontekstu. Zabezpieczenie realizowane przez restrykcyjny System Prompt.
*   **Obsługa zapytań Out-of-Domain:** Jeśli pytanie nie dotyczy Konstytucji lub brakuje na nie odpowiedzi w wyszukanych chunkach, system musi odpowiedzieć z góry ustaloną formułą (np. "Konstytucja RP nie reguluje tej kwestii.").
*   **Wydajność:** System powinien działać płynnie na lokalnej maszynie z GPU posiadającym do 16GB VRAM.
*   **Modularność:** Logika RAG (połączenie z Qdrant, generowanie embeddingów, komunikacja z LLM) musi być oddzielona od warstwy widoku (Streamlit).

## 5. Historie Użytkownika (User Stories)
1.  *Jako obywatel, chcę zapytać "Kto uchwala budżet państwa?", aby dowiedzieć się, jaki organ jest za to odpowiedzialny według Konstytucji.*
2.  *Jako student prawa, chcę zapytać "Jakie są warunki wprowadzenia stanu wyjątkowego?", aby otrzymać precyzyjną odpowiedź z powołaniem na konkretne artykuły.*
3.  *Jako użytkownik testujący system, chcę zapytać "Jaki jest przepis na sernik?", aby sprawdzić, czy system odmówi odpowiedzi zgodnie z założeniami (Out-of-Domain).*

## 6. Kryteria Sukcesu
*   Aplikacja uruchamia się lokalnie bez błędów.
*   Baza Qdrant poprawnie przechowuje wszystkie 243 artykuły Konstytucji.
*   System poprawnie identyfikuje i zwraca odpowiednie artykuły dla zapytań ustrojowych.
*   Model Bielik generuje poprawne językowo odpowiedzi w j. polskim, nie zmyśla informacji i zawsze podaje źródło.
*   Wszystkie kluczowe akcje i błędy są poprawnie zapisywane w pliku logów przez `loguru`.