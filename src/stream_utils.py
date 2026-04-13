from typing import Generator, Iterable
from loguru import logger


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
            logger.info("Odpowiedź została wygenerowana pomyślnie.")
            if (
                hasattr(chunk, "response")
                and hasattr(chunk.response, "usage")
                and chunk.response.usage
            ):
                usage = chunk.response.usage
                usage_stats["input"] = getattr(usage, "input_tokens", 0)
                usage_stats["output"] = getattr(usage, "output_tokens", 0)
