from typing import Generator, Iterable
from loguru import logger

# Cache przechowuje gotowe odpowiedzi LLM, kluczowane (query, temperature, top_k).
# Wartość: (answer_text, sources, usage_stats) — usage_stats pozwala logować
# ile tokenów zaoszczędzono dzięki trafieniu w cache przy kolejnych wywołaniach.
# Jako zmienna modułowa persystuje przez cały czas życia procesu Streamlit —
# nie resetuje się przy rerenderach skryptu (w przeciwieństwie do zmiennych w app.py).
_answer_cache: dict[tuple[str, float, int], tuple[str, list[str], dict[str, int]]] = {}


def get_cached_answer(
    query: str,
    temperature: float,
    top_k: int
) -> tuple[str, list[str], dict[str, int]] | None:
    """Zwraca (answer_text, sources, usage_stats) z cache lub None jeśli brak trafienia."""
    return _answer_cache.get((query, temperature, top_k))


def cache_answer(
    query: str,
    temperature: float,
    top_k: int,
    answer_text: str,
    sources: list[str],
    usage_stats: dict[str, int]
) -> None:
    """Zapisuje odpowiedź wraz ze statystykami tokenów do cache."""
    _answer_cache[(query, temperature, top_k)] = (answer_text, sources, usage_stats)


def fake_stream(text: str) -> Generator[str, None, None]:
    """
    Symuluje efekt strumieniowania dla odpowiedzi pobranych z cache.
    Zwraca tekst słowo po słowie — zachowuje wizualny efekt pisania
    bez ponownego wywołania modelu LLM.
    """
    words = text.split(" ")
    for i, word in enumerate(words):
        # Dodajemy spację po każdym słowie poza ostatnim
        yield word + (" " if i < len(words) - 1 else "")


def parse_stream(
    response_stream: Iterable,
    usage_stats: dict[str, int]
) -> Generator[str, None, None]:
    """
    Generator parsujący strumień zdarzeń Responses API.
    Wyciąga fragmenty tekstu i aktualizuje usage_stats po zakończeniu.
    """
    for chunk in response_stream:
        # 1. Wyłapujemy paczki z tekstem
        if getattr(chunk, "type", None) == "response.output_text.delta":
            yield chunk.delta
        # 2. Pobieramy statystyki ze zdarzenia końcowego response.completed
        elif getattr(chunk, "type", None) == "response.completed":
            if hasattr(chunk, "response"):
                response = chunk.response
                usage = getattr(response, "usage", None)
                if usage:
                    input_tokens = getattr(usage, "input_tokens", 0)
                    output_tokens = getattr(usage, "output_tokens", 0)
                    usage_stats["input"] = input_tokens
                    usage_stats["output"] = output_tokens
                    cached_tokens = getattr(
                        getattr(usage, "input_tokens_details", None),
                        "cached_tokens", 0
                    )
                    logger.info(
                        "Odpowiedź została wygenerowana pomyślnie. | "
                        f"model: {getattr(response, 'model', 'N/A')} | "
                        f"id: {getattr(response, 'id', 'N/A')} | "
                        f"temperature: {getattr(response, 'temperature', 'N/A')} | "
                        f"input: {input_tokens} tokens (cached: {cached_tokens}), "
                        f"output: {output_tokens} tokens."
                    )
