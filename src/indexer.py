import warnings
from typing import Callable
from fastembed.common.types import NumpyArray
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import TextEmbedding
from loguru import logger
from chunker import parse_constitution
from config_loader import cfg

# Stałe
COLLECTION_NAME = cfg["embedding"]["collection_name"]
EMBEDDING_MODEL = cfg["embedding"]["model_name"]
QDRANT_DB_PATH = cfg["paths"]["db_path"]
KONSTYTUCJA_PATH = cfg["paths"]["data_file"]
VECTOR_SIZE = cfg["embedding"]["vector_size"]


def create_index(
    progress_callback: Callable[[int, int], None] | None = None
) -> None:
    # 1. Pobranie chunków z naszego poprzedniego skryptu
    logger.info("Pobieranie fragmentów Konstytucji...")
    chunks = parse_constitution(KONSTYTUCJA_PATH)

    if not chunks:
        logger.error("Brak danych do indeksowania. Sprawdź plik źródłowy.")
        return

    # 2. Inicjalizacja modelu do embeddingów
    # (pobierze się automatycznie przy pierwszym uruchomieniu)
    logger.info(f"Ładowanie modelu embeddingów: {EMBEDDING_MODEL}...")
    # Wymiar dla e5-large to 1024;
    # fastembed>=0.8 używa mean poolingu (ostrzeżenie o CLS dotyczy migracji)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"The model .* now uses mean pooling instead of CLS embedding",
            category=UserWarning,
        )
        embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL)

    # 3. Inicjalizacja lokalnej bazy Qdrant
    logger.info("Inicjalizacja bazy Qdrant...")
    client = QdrantClient(path=QDRANT_DB_PATH)

    # Tworzenie nowej kolekcji (jeśli istnieje, usuwamy ją i tworzymy na nowo)
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
        logger.info(f"Usunięto starą kolekcję: {COLLECTION_NAME}")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE,  # Wymiar wektora dla modelu e5-large
            distance=models.Distance.COSINE  # Miara odległości (Cosinusowa)
        )
    )

    # 4. Przygotowanie danych do wektoryzacji
    # Dodajemy prefix "passage: " wymagany przez modele E5
    # Dołączamy też nazwę rozdziału do tekstu, aby model rozumiał kontekst
    documents = [
        f"passage: Rozdział: {chunk['chapter']}\n{chunk['text']}"
        for chunk in chunks
    ]

    # 5. Generowanie embeddingów
    logger.info("Generowanie wektorów.")
    total = len(documents)
    embeddings_list: list[NumpyArray] = []
    for i, emb in enumerate(embedding_model.embed(documents)):
        embeddings_list.append(emb)
        if progress_callback:
            progress_callback(i + 1, total)

    # 6. Zapis do bazy Qdrant
    logger.info("Zapisywanie danych do bazy Qdrant...")

    # Przygotowanie punktów (wektor + metadane)
    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings_list)):
        points.append(
            models.PointStruct(
                id=i,
                vector=embedding.tolist(),
                payload={
                    "article_num": chunk["article_num"],
                    "chapter": chunk["chapter"],
                    "text": chunk["text"]
                }
            )
        )

    # Wgrywanie paczkami (batch)
    client.upload_points(
        collection_name=COLLECTION_NAME,
        points=points
    )

    logger.info(
        "Sukces! Zapisano "
        f"{len(points)} fragmentów do bazy Qdrant w folderze {QDRANT_DB_PATH}."
    )


if __name__ == "__main__":
    create_index()
