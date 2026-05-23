from __future__ import annotations

import argparse
import random
import re
import string
import sys
import time
from typing import Callable, TextIO

from wordfreq import zipf_frequency

from . import __version__

TITLE = "(Not quite infinite monkeys (only 1 actually) Typing)."
DESCRIPTION = (
    "This program simulates a single monkey randomly typing on a standard "
    "QWERTY keyboard."
)
CHARACTER_POOL = string.ascii_letters + " " + string.punctuation
LOWERCASE_POOL = string.ascii_lowercase + " "
UPPERCASE_POOL = string.ascii_uppercase + " "
MIXED_CASE_POOL = string.ascii_letters + " "
ALL_CHARACTER_POOL = CHARACTER_POOL
ENGLISH_WORD_ZIPF_THRESHOLD = 3.8


def get_character_pool(mode: int) -> str:
    if mode == 1:
        return LOWERCASE_POOL
    if mode == 2:
        return UPPERCASE_POOL
    if mode == 3:
        return MIXED_CASE_POOL
    return ALL_CHARACTER_POOL


def prompt_character_mode() -> int:
    print("Choose the character set:")
    print("1. Lowercase letters and spaces")
    print("2. Uppercase letters and spaces")
    print("3. Uppercase and lowercase letters and spaces")
    print("4. All characters")

    while True:
        raw_value = input("Enter 1, 2, 3, or 4: ").strip()
        try:
            selection = int(raw_value)
        except ValueError:
            print("Please enter 1, 2, 3, or 4.")
            continue

        if selection in {1, 2, 3, 4}:
            return selection

        print("Please enter 1, 2, 3, or 4.")


def _is_english_word(token: str) -> bool:
    lowered_token = token.lower()
    if len(lowered_token) < 3:
        return False

    if not lowered_token.isalpha():
        return False

    return zipf_frequency(lowered_token, "en") >= ENGLISH_WORD_ZIPF_THRESHOLD


def _is_content_word(token: str) -> bool:
    return _is_english_word(token) and len(token) >= 3


def prompt_character_count() -> int:
    while True:
        raw_value = input("How many characters should the monkey type? [100-1000000]: ").strip()
        try:
            count = int(raw_value)
        except ValueError:
            print("Please enter a whole number between 100 and 1000000.")
            continue

        if 100 <= count <= 1_000_000:
            return count

        print("Please enter a whole number between 100 and 1000000.")


def generate_monkey_text(
    character_count: int,
    character_pool: str = CHARACTER_POOL,
    stream: TextIO | None = None,
    newline_interval: int = 80,
    start_index: int = 0,
    flush_interval: int = 80,
    typing_delay: float = 0.0,
    random_choice: Callable[[str], str] = random.choice,
) -> str:
    output_stream = stream or sys.stdout
    generated_characters: list[str] = []

    for index in range(character_count):
        character = random_choice(character_pool)
        generated_characters.append(character)
        output_stream.write(character)

        absolute_position = start_index + index + 1
        is_line_break = newline_interval > 0 and absolute_position % newline_interval == 0
        if is_line_break:
            output_stream.write("\n")

        if is_line_break or (flush_interval > 0 and (index + 1) % flush_interval == 0):
            output_stream.flush()

        if typing_delay > 0:
            time.sleep(typing_delay)

    absolute_end = start_index + character_count
    if newline_interval > 0 and absolute_end % newline_interval != 0:
        output_stream.write("\n")
    output_stream.flush()

    return "".join(generated_characters)


def analyze_text(text: str) -> tuple[list[str], list[str], list[str]]:
    candidate_words = re.findall(r"[A-Za-z]{2,}", text)
    words = [word for word in candidate_words if _is_english_word(word)]

    phrase_candidates: list[str] = []
    for match in re.finditer(r"(?:[A-Za-z]{2,}\s+){1,6}[A-Za-z]{2,}", text):
        phrase = " ".join(match.group(0).split())
        phrase_words = phrase.split()
        if (
            len(phrase_words) >= 2
            and all(_is_english_word(word) for word in phrase_words)
            and any(_is_content_word(word) for word in phrase_words)
        ):
            phrase_candidates.append(phrase)

    sentence_candidates: list[str] = []
    for match in re.finditer(r"[^.!?]{8,}[.!?]", text):
        sentence = " ".join(match.group(0).split())
        sentence_words = re.findall(r"[A-Za-z]{2,}", sentence)
        if (
            len(sentence_words) >= 2
            and all(_is_english_word(word) for word in sentence_words)
            and any(_is_content_word(word) for word in sentence_words)
        ):
            sentence_candidates.append(sentence)

    unique_words = list(dict.fromkeys(words))
    unique_phrases = list(dict.fromkeys(phrase_candidates))
    unique_sentences = list(dict.fromkeys(sentence_candidates))
    return unique_words, unique_phrases, unique_sentences


def run_batched_simulation(
    character_count: int,
    character_pool: str,
    batch_size: int = 10_000,
    stream: TextIO | None = None,
    newline_interval: int = 80,
    flush_interval: int = 80,
    typing_delay: float = 0.0,
    random_choice: Callable[[str], str] = random.choice,
    analyzer: Callable[[str], tuple[list[str], list[str], list[str]]] = analyze_text,
    show_batch_progress: bool = True,
    analysis_overlap: int = 64,
) -> tuple[list[str], list[str], list[str], float, float]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    if analysis_overlap < 0:
        raise ValueError("analysis_overlap cannot be negative")

    unique_words: dict[str, None] = {}
    unique_phrases: dict[str, None] = {}
    unique_sentences: dict[str, None] = {}
    generation_seconds = 0.0
    analysis_seconds = 0.0
    processed = 0
    carry_text = ""

    while processed < character_count:
        current_batch_size = min(batch_size, character_count - processed)

        generation_start = time.perf_counter()
        chunk_text = generate_monkey_text(
            current_batch_size,
            character_pool=character_pool,
            stream=stream,
            newline_interval=newline_interval,
            start_index=processed,
            flush_interval=flush_interval,
            typing_delay=typing_delay,
            random_choice=random_choice,
        )
        generation_seconds += time.perf_counter() - generation_start

        analysis_start = time.perf_counter()
        analysis_text = carry_text + chunk_text
        words, phrases, sentences = analyzer(analysis_text)
        analysis_seconds += time.perf_counter() - analysis_start

        for word in words:
            unique_words.setdefault(word, None)
        for phrase in phrases:
            unique_phrases.setdefault(phrase, None)
        for sentence in sentences:
            unique_sentences.setdefault(sentence, None)

        processed += current_batch_size

        # Explicitly release this batch text before processing the next one.
        if analysis_overlap > 0:
            carry_text = chunk_text[-analysis_overlap:]
        else:
            carry_text = ""
        del chunk_text

        if show_batch_progress and processed < character_count:
            print()
            print(f"Processed {processed:,}/{character_count:,} characters...")

    return (
        list(unique_words),
        list(unique_phrases),
        list(unique_sentences),
        generation_seconds,
        analysis_seconds,
    )


def display_results(words: list[str], phrases: list[str], sentences: list[str]) -> None:
    print()
    print("Results")
    print("-------")

    if words:
        print("Possible words:")
        for word in words[:25]:
            print(f"- {word}")
    else:
        print("Possible words: none found.")

    print()

    if phrases:
        print("Possible phrases:")
        for phrase in phrases[:15]:
            print(f"- {phrase}")
    else:
        print("Possible phrases: none found.")

    print()

    if sentences:
        print("Possible sentences:")
        for sentence in sentences[:10]:
            print(f"- {sentence}")
    else:
        print("Possible sentences: none found.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="monkeys-typing",
        description="Monkeys Typing command-line app",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"Monkeys Typing {__version__}",
    )
    return parser


def main() -> int:
    build_parser().parse_args()

    print(TITLE)
    print(DESCRIPTION)
    print()

    character_count = prompt_character_count()
    character_mode = prompt_character_mode()
    character_pool = get_character_pool(character_mode)
    batch_size = 10_000

    print()
    print(f"Typing {character_count} characters...")
    print(f"Running in batches of {batch_size:,} characters.")
    print()

    words, phrases, sentences, generation_seconds, analysis_seconds = run_batched_simulation(
        character_count=character_count,
        character_pool=character_pool,
        batch_size=batch_size,
    )

    print(f"Analysis complete in {analysis_seconds:.2f}s (generation: {generation_seconds:.2f}s).")
    display_results(words, phrases, sentences)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())