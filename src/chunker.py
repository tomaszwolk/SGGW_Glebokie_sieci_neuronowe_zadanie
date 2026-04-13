import re
from loguru import logger
from config_loader import cfg

KONSTYTUCJA_PATH = cfg["paths"]["data_file"]


def parse_constitution(file_path: str) -> list[dict]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        logger.error(f"Nie znaleziono pliku: {file_path}")
        return []

    chunks: list[dict] = []
    current_chapter: str = "Preambuła"
    current_subchapter: str = ""
    current_article_num: str = "Preambuła"
    current_article_text: list[str] = []

    # KROK 1: Czyszczenie tekstu z potrójnych grawisów (```)
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped == "```":
            continue  # Pomijamy linie zawierające tylko grawisy
        cleaned_lines.append(stripped)

    # KROK 2: Parsowanie linia po linii
    i = 0
    while i < len(cleaned_lines):
        line = cleaned_lines[i]

        if not line:
            i += 1
            continue

        # Wykrywanie Rozdziału (np. "Rozdział I")
        if line.startswith("Rozdział"):
            current_chapter = line
            # Resetujemy podrozdział przy nowym rozdziale
            current_subchapter = ""
            i += 1
            # Pobieramy tytuł rozdziału (następna niepusta linia)
            while i < len(cleaned_lines) and not cleaned_lines[i]:
                i += 1
            if (
                i < len(cleaned_lines)
                and not cleaned_lines[i].startswith("Art.")
                and not cleaned_lines[i].startswith("Rozdział")
            ):
                current_chapter += " - " + cleaned_lines[i]
            i += 1
            continue

        # Wykrywanie podrozdziału (np. "### ZASADY OGÓLNE")
        if line.startswith("### "):
            current_subchapter = line.replace("### ", "").strip()
            i += 1
            continue

        # Wykrywanie Artykułu (np. "Art. 1.")
        article_match = re.match(r"^Art\.\s*(\d+[a-z]*)\.", line)
        if article_match:
            # Zapisujemy poprzedni artykuł (lub preambułę) do listy chunks
            if current_article_text:
                text_joined = "\n".join(current_article_text).strip()

                if text_joined:
                    # Budujemy pełną nazwę rozdziału z podrozdziałem
                    chapter_meta = current_chapter
                    if current_subchapter:
                        chapter_meta += f" ({current_subchapter})"

                    chunks.append(
                        {
                            "article_num": current_article_num,
                            "chapter": chapter_meta,
                            "text": text_joined,
                        }
                    )

            # Zaczynamy zbierać nowy artykuł
            current_article_num = article_match.group(1)
            current_article_text = [line]
        else:
            # Zwykły tekst - dodajemy do obecnego artykułu/preambuły
            # Pomijamy główne nagłówki na samej górze dla czystości
            if line not in [
                "# KONSTYTUCJA",
                "## RZECZYPOSPOLITEJ POLSKIEJ",
            ] and not line.startswith("z dnia 2 kwietnia"):
                current_article_text.append(line)

        i += 1

    # Dodanie ostatniego artykułu po zakończeniu pętli
    if current_article_text:
        chapter_meta = current_chapter
        if current_subchapter:
            chapter_meta += f" ({current_subchapter})"
        chunks.append(
            {
                "article_num": current_article_num,
                "chapter": chapter_meta,
                "text": "\n".join(current_article_text).strip(),
            }
        )

    logger.info(
        "Zakończono parsowanie. "
        f"Znaleziono {len(chunks)} fragmentów (w tym Preambuła)."
    )
    return chunks


if __name__ == "__main__":
    # Wyświetlamy dla testów
    chunks = parse_constitution(KONSTYTUCJA_PATH)

    if chunks:
        print(f"Liczba wyciągniętych chunków: {len(chunks)}\n")

        # Wyświetlmy Preambułę (zazwyczaj pierwszy element)
        print("--- CHUNK 1 ---")
        print(f"Artykuł: {chunks[0]['article_num']}")
        print(f"Rozdział: {chunks[0]['chapter']}")
        print(f"Tekst:\n{chunks[0]['text'][:150]}...\n")

        # Wyświetlmy przykładowy artykuł z podrozdziałem
        for chunk in chunks:
            if chunk["article_num"] == "30":
                print("--- CHUNK Art. 30 ---")
                print(f"Artykuł: {chunk['article_num']}")
                print(f"Rozdział: {chunk['chapter']}")
                print(f"Tekst:\n{chunk['text']}")
                break
