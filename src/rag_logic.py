import atexit
import warnings

from openai import OpenAI
from qdrant_client import QdrantClient
from fastembed import TextEmbedding
from loguru import logger
from config_loader import cfg

# Stałe konfiguracyjne
QDRANT_DB_PATH = cfg["paths"]["db_path"]
COLLECTION_NAME = cfg["embedding"]["collection_name"]
EMBEDDING_MODEL = cfg["embedding"]["model_name"]
SYSTEM_PROMPT = cfg["llm"]["system_prompt"]
TOP_K = cfg["embedding"]["top_k"]

# Konfiguracja LM Studio
LM_STUDIO_URL = cfg["llm"]["base_url"]
LLM_MODEL_NAME = cfg["llm"]["model_name"]
LLM_API_KEY = cfg["llm"]["api_key"]
LLM_TEMPERATURE = cfg["llm"]["temperature"]


class RAGInitError(Exception):
    """
    Wyjątek dla błędów podczas inicjalizacji komponentów RAG.
    """
    pass


# Inicjalizacja klientów (robimy to raz, globalnie dla modułu)
logger.info("Inicjalizacja komponentów RAG...")
try:
    qdrant_client = QdrantClient(path=QDRANT_DB_PATH)
    # Zamknięcie przed shutdownem interpretera
    # — inaczej __del__ woła close() gdy importy już nie działają
    atexit.register(qdrant_client.close)
except Exception as e:
    logger.error(f"Błąd podczas inicjalizacji klienta Qdrant: {e}")
    raise RAGInitError(
        "Nie można otworzyć bazy wektorowej Qdrant. "
        "Sprawdź czy ścieżka jest poprawna lub usuń folder bazy i uruchom ponownie."
    ) from e

try:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"The model .* now uses mean pooling instead of CLS embedding",
            category=UserWarning,
        )
        embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL)
except Exception as e:
    logger.error(f"Błąd podczas inicjalizacji modelu embeddingów: {e}")
    raise RAGInitError(
        f"Nie można załadować modelu embeddingów: {EMBEDDING_MODEL}. "
        "Sprawdź połączenie z internetem lub zmień model w pliku konfiguracyjnym."
    ) from e

try:
    llm_client = OpenAI(base_url=LM_STUDIO_URL, api_key=LLM_API_KEY)
except Exception as e:
    logger.error(f"Błąd podczas inicjalizacji komponentów: {e}")
    raise RAGInitError(
        f"Nie można połączyć się z LM Studio: {LM_STUDIO_URL}. "
        "Sprawdź czy jest uruchomione i załadowany model."
    ) from e


def retrieve_context(query: str, top_k: int = TOP_K) -> tuple[str, list[str]]:
    """
    Wyszukuje najbardziej pasujące fragmenty w bazie Qdrant.
    Zwraca połączony tekst kontekstu oraz listę źródeł.
    """
    logger.info(f"Wyszukiwanie kontekstu dla zapytania: '{query}' z top_k: {top_k}")

    # Model E5 wymaga przedrostka "query: " dla zapytań
    query_for_embedding = f"query: {query}"

    # Generowanie wektora dla pytania (embed zwraca Iterable — next() wymaga iteratora)
    query_vector = next(iter(embedding_model.query_embed(query_for_embedding))).tolist()

    # Wyszukiwanie w Qdrant
    search_results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
    )

    context_parts = []
    sources = []

    for hit in search_results.points:
        payload = hit.payload or {}
        art_num = payload.get("article_num", "Nieznany")
        chapter = payload.get("chapter", "Nieznany")
        text = payload.get("text", "")

        # Budowanie czytelnego bloku tekstu dla LLM
        context_parts.append(f"--- {chapter}, Art. {art_num} ---\n{text}")
        sources.append(f"Art. {art_num}")

    full_context = "\n\n".join(context_parts)
    logger.debug(f"Znalezione źródła: {sources}")

    return full_context, sources


def stream_answer(
    query: str,
    context: str,
    temperature: float = LLM_TEMPERATURE
):
    """
    Wysyła prompt do lokalnego modelu LLM przez LM Studio (Responses API).
    """
    # Łączymy kontekst z pytaniem użytkownika
    user_input = (
        f"Kontekst z Konstytucji RP:\n{context}\n\n"
        f"Pytanie użytkownika: {query}"
    )

    logger.info("Wysyłanie zapytania do modelu LLM (LM Studio)...")
    try:
        return llm_client.responses.create(
            model=LLM_MODEL_NAME,
            input=user_input,
            instructions=SYSTEM_PROMPT,
            stream=True,
            temperature=temperature
        )
    except Exception as e:
        logger.error(f"Błąd komunikacji z LM Studio: {e}")
        return None


def ask_constitution_stream(
    query: str,
    temperature: float = LLM_TEMPERATURE,
    top_k: int = TOP_K
) -> tuple:
    """
    Główna funkcja spinająca cały pipeline RAG. Zwraca strumień odpowiedzi.
    """
    logger.info(f"Nowe zapytanie: {query}")

    # 1. Pobranie kontekstu
    context, sources = retrieve_context(query, top_k=top_k)

    # 2. Generowanie odpowiedzi
    response_stream = stream_answer(query, context, temperature)

    return response_stream, sources


if __name__ == "__main__":
    # Prosty test w terminalu
    print("--- Testowanie logiki RAG ---")
    test_query = "Kto może zostać prezydentem?"
    print(f"Pytanie: {test_query}\n")

    response_stream, sources = ask_constitution_stream(test_query)

    print(f"Źródła: {', '.join(sources)}")
    if response_stream:
        for chunk in response_stream:
            if chunk.type == "response.output_text.delta":
                print(chunk.delta, end="", flush=True)
    else:
        print("Nie udało się uzyskać odpowiedzi z LM Studio.")
