import streamlit as st
from pathlib import Path
from loguru import logger
from config_loader import cfg
from stream_utils import parse_stream, fake_stream, get_cached_answer, cache_answer
from streamlit.delta_generator import DeltaGenerator
from functools import partial

MAX_CONTEXT_TOKENS = cfg["llm"]["max_session_tokens"]


def on_progress(
    current: int,
    total: int,
    progress_bar: DeltaGenerator
) -> None:
    progress_bar.progress(
        current / total,
        text=f"Generowanie embeddingów: {current}/{total}"
    )


# Konfiguracja loguru dla UI
@st.cache_resource
def _setup_logger():
    return logger.add(cfg["paths"]["log_file"], rotation="1 MB", level="INFO")


_setup_logger()

# Konfiguracja strony Streamlit
st.set_page_config(
    page_title=cfg["ui"]["title"],
    page_icon=cfg["ui"]["page_icon"],
    layout=cfg["ui"]["layout"]
)


# Automatyczne tworzenie bazy wiedzy, jeśli nie istnieje
db_path = Path(cfg["paths"]["db_path"])
collection_name = cfg["embedding"]["collection_name"]
sqlite_path = db_path / "collection" / collection_name / "storage.sqlite"


def collection_has_data() -> bool:
    try:
        return sqlite_path.exists() and sqlite_path.stat().st_size > 100_000
    except Exception:
        return False


if not collection_has_data():
    logger.info("Baza wektorowa nie istnieje — uruchamiam indeksowanie...")
    from indexer import create_index

    with st.status("Pierwsza konfiguracja bazy wiedzy...", expanded=True) as status:
        st.write("Parsowanie Konstytucji RP...")
        # Tworzymy pusty pasek postępu
        progress_bar = st.progress(0, text="Generowanie embeddingów...")

        st.write("Generowanie embeddingów... (Może to zająć kilka minut...)")
        create_index(progress_callback=partial(on_progress, progress_bar=progress_bar))

        progress_bar.empty()  # Usuwa pasek postępu po zakończeniu
        status.update(label="Baza wiedzy gotowa!", state="complete", expanded=False)

    st.rerun()

# Import z rag_logic musi być po set_page_config
try:
    from rag_logic import ask_constitution_stream
except Exception as e:
    if type(e).__name__ == "RAGInitError":
        st.error(str(e))
        st.stop()
    raise

st.title(cfg["ui"]["title"])
st.markdown(cfg["ui"]["description"])

# Inicjalizacja historii czatu w sesji Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_input_tokens" not in st.session_state:
    st.session_state.total_input_tokens = 0
if "total_output_tokens" not in st.session_state:
    st.session_state.total_output_tokens = 0

with st.sidebar:
    st.header("O projekcie")
    st.info(f"System RAG oparty o model {cfg['llm']['model_name']}.")
    st.header("⚙️ Ustawienia modelu")
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=cfg["llm"]["temperature"],
        step=0.1,
        help="Temperature - parametr kontrolujący losowość odpowiedzi modelu."
    )
    top_k = st.slider(
        "Top K",
        min_value=1,
        max_value=10,
        value=cfg["embedding"]["top_k"],
        step=1,
        help="Top K - liczba najbardziej pasujących fragmentów do wyszukania."
    )
    st.divider()  # Linia oddzielająca sekcje
    st.header("📊 Statystyki sesji")
    st.metric("Input Tokens (Suma)", st.session_state.total_input_tokens)
    st.metric("Output Tokens (Suma)", st.session_state.total_output_tokens)

    total = st.session_state.total_input_tokens + st.session_state.total_output_tokens
    st.write(f"**Łącznie:** {total} / {MAX_CONTEXT_TOKENS} tokens")

    # Pasek postępu do limitu (opcjonalnie)
    progress = min(total / MAX_CONTEXT_TOKENS, 1.0)
    st.progress(progress, text="Zużycie limitu sesji")

    if st.button("Wyczyść czat i statystyki"):
        st.session_state.messages = []
        st.session_state.total_input_tokens = 0
        st.session_state.total_output_tokens = 0
        st.rerun()

# Wyświetlanie historii czatu
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Jeśli asystent podał źródła, wyświetlamy je jako mały tekst
        if "sources" in message and message["sources"]:
            st.caption(f"Źródła: {', '.join(message['sources'])}")

# Pole do wprowadzania zapytań przez użytkownika
if prompt := st.chat_input("Zadaj pytanie. Przykład: Kto wypowiada wojnę?"):

    # 1. Dodanie pytania użytkownika do historii i wyświetlenie go
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # 2. Generowanie odpowiedzi asystenta
    with st.chat_message("assistant"):
        cached = get_cached_answer(prompt, temperature, top_k)

        if cached is not None:
            # --- ŚCIEŻKA CACHE HIT ---
            # Identyczne zapytanie było już zadane w tej sesji — zwracamy
            # zapamiętaną odpowiedź bez wywołania LLM. Efekt pisania zachowany
            # dzięki fake_stream. Tokeny nie są zużywane.
            cached_text, sources, saved_stats = cached
            logger.info(
                f"Cache hit dla zapytania: '{prompt}' | "
                f"zaoszczędzono: {saved_stats['input']} input "
                f"+ {saved_stats['output']} output tokens"
            )
            st.write_stream(fake_stream(cached_text))
            answer_text = cached_text
            usage_stats = {"input": 0, "output": 0}

        else:
            # --- ŚCIEŻKA CACHE MISS ---
            # Pierwsze zapytanie z tym kluczem — wywołujemy LLM i streamujemy
            # odpowiedź w czasie rzeczywistym, po czym zapisujemy do cache.
            stream, sources = ask_constitution_stream(prompt, temperature, top_k)

            if not stream:
                st.error(
                    "Nie można połączyć się z LM Studio. "
                    "Upewnij się, że jest uruchomione i załadowany model."
                )
                st.stop()

            # Pusty słownik na statystyki wyłapywane w locie przez parse_stream
            usage_stats = {"input": 0, "output": 0}

            # st.write_stream strumieniuje tekst na ekran słowo po słowie
            # i zwraca pełny tekst (str) albo listę chunków — typ: str | list[Any]
            full_answer = st.write_stream(parse_stream(stream, usage_stats))
            answer_text = (
                full_answer
                if isinstance(full_answer, str)
                else "".join(str(x) for x in full_answer)
            )
            logger.info(
                f"Odpowiedź: {answer_text[:500]}"
                f"{'...' if len(answer_text) > 500 else ''} | "
                f"Źródła: {', '.join(sources)}"
            )

            # --- LOGIKA FILTROWANIA ŹRÓDEŁ ---
            # Jeśli odpowiedź nie zawiera "art" i pasuje do frazy odmownej,
            # czyścimy listę źródeł — model stwierdził brak regulacji w Konstytucji.
            if "art" not in answer_text.lower():
                negative_responses = [
                    "konstytucja rp nie reguluje",
                    "nie reguluje tej kwestii",
                    "nie reguluje kwestii"
                ]
                if any(phrase in answer_text.lower() for phrase in negative_responses):
                    sources = []

            # Zapisujemy do cache po odfiltrowaniu źródeł — kolejne identyczne
            # zapytania dostaną już gotowy wynik bez wywołania LLM.
            # usage_stats.copy() zapobiega mutacji zapisanych wartości przez późniejszy kod.
            cache_answer(prompt, temperature, top_k, answer_text, sources, usage_stats.copy())

        # Wyświetlamy źródła (wspólne dla obu ścieżek)
        if sources:
            st.caption(f"Źródła: {', '.join(sources)}")

        # Aktualizacja statystyk sesji (dla cache hit usage_stats = {0, 0})
        st.session_state.total_input_tokens += usage_stats["input"]
        st.session_state.total_output_tokens += usage_stats["output"]

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer_text,
            "sources": sources
        })

        # Wymuszamy odświeżenie paska bocznego (statystyki tokenów)
        st.rerun()
