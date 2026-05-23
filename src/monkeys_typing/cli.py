from __future__ import annotations

import argparse
import base64
import os
import random
import re
import string
import sys
import time
import json
import pickle
from pathlib import Path
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
MIN_CHARACTER_COUNT = 100
MAX_CHARACTER_COUNT = 1_000_000_000
CHECKPOINT_VERSION = 1
CHECKPOINT_BATCH_INTERVAL = 10


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
        raw_value = input(
            f"How many characters should the monkey type? [{MIN_CHARACTER_COUNT}-{MAX_CHARACTER_COUNT}]: "
        ).strip()
        try:
            count = int(raw_value)
        except ValueError:
            print(
                f"Please enter a whole number between {MIN_CHARACTER_COUNT} and {MAX_CHARACTER_COUNT}."
            )
            continue

        if MIN_CHARACTER_COUNT <= count <= MAX_CHARACTER_COUNT:
            return count

        print(f"Please enter a whole number between {MIN_CHARACTER_COUNT} and {MAX_CHARACTER_COUNT}.")


def prompt_run_mode() -> int:
    print("Choose run mode:")
    print("1. Display generated characters in terminal")
    print("2. Fast mode (no character rendering)")
    print("3. Fast mode with output file")

    while True:
        raw_value = input("Enter 1, 2, or 3: ").strip()
        try:
            selection = int(raw_value)
        except ValueError:
            print("Please enter 1, 2, or 3.")
            continue

        if selection in {1, 2, 3}:
            return selection

        print("Please enter 1, 2, or 3.")


def prompt_yes_no(prompt: str, default_yes: bool = True) -> bool:
    suffix = "[Y/n]" if default_yes else "[y/N]"
    while True:
        raw_value = input(f"{prompt} {suffix}: ").strip().lower()
        if not raw_value:
            return default_yes
        if raw_value in {"y", "yes"}:
            return True
        if raw_value in {"n", "no"}:
            return False
        print("Please answer with y or n.")


def prompt_checkpoint_path() -> Path:
    default_path = "monkey_checkpoint.json"
    raw_value = input(f"Checkpoint file path [default: {default_path}]: ").strip()
    if not raw_value:
        return Path(default_path)
    return Path(raw_value)


def _serialize_random_state(state: object) -> str:
    return base64.b64encode(pickle.dumps(state)).decode("ascii")


def _deserialize_random_state(encoded_state: str) -> object:
    return pickle.loads(base64.b64decode(encoded_state.encode("ascii")))


def load_checkpoint(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as checkpoint_file:
        data = json.load(checkpoint_file)

    if data.get("version") != CHECKPOINT_VERSION:
        print("Checkpoint version mismatch; starting a fresh run.")
        return None

    return data


def save_checkpoint(path: Path, checkpoint_data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    last_error: OSError | None = None

    for attempt in range(5):
        try:
            with temp_path.open("w", encoding="utf-8") as checkpoint_file:
                json.dump(checkpoint_data, checkpoint_file)
            temp_path.replace(path)
            return
        except OSError as error:
            last_error = error
            time.sleep(0.1 * (attempt + 1))
        finally:
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except OSError:
                pass

    # Final fallback for transient file-lock scenarios: direct overwrite.
    try:
        with path.open("w", encoding="utf-8") as checkpoint_file:
            json.dump(checkpoint_data, checkpoint_file)
        return
    except OSError as error:
        last_error = error

    if last_error is not None:
        raise last_error


def prompt_output_file_path() -> Path:
    default_path = "monkey_output.txt"
    raw_value = input(
        f"Output file path for generated text [default: {default_path}]: "
    ).strip()

    if not raw_value:
        return Path(default_path)

    return Path(raw_value)


def generate_monkey_text(
    character_count: int,
    character_pool: str = CHARACTER_POOL,
    stream: TextIO | None = None,
    render_output: bool = True,
    newline_interval: int = 80,
    start_index: int = 0,
    flush_interval: int = 80,
    typing_delay: float = 0.0,
    random_choice: Callable[[str], str] = random.choice,
) -> str:
    output_stream = stream if stream is not None else (sys.stdout if render_output else None)
    generated_characters: list[str] = []

    for index in range(character_count):
        character = random_choice(character_pool)
        generated_characters.append(character)
        if output_stream is not None:
            output_stream.write(character)

        absolute_position = start_index + index + 1
        is_line_break = newline_interval > 0 and absolute_position % newline_interval == 0
        if output_stream is not None and is_line_break:
            output_stream.write("\n")

        if output_stream is not None and (
            is_line_break or (flush_interval > 0 and (index + 1) % flush_interval == 0)
        ):
            output_stream.flush()

        if typing_delay > 0:
            time.sleep(typing_delay)

    absolute_end = start_index + character_count
    if output_stream is not None and newline_interval > 0 and absolute_end % newline_interval != 0:
        output_stream.write("\n")
    if output_stream is not None:
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
    render_output: bool = True,
    newline_interval: int = 80,
    flush_interval: int = 80,
    typing_delay: float = 0.0,
    random_choice: Callable[[str], str] = random.choice,
    analyzer: Callable[[str], tuple[list[str], list[str], list[str]]] = analyze_text,
    show_batch_progress: bool = True,
    analysis_overlap: int = 64,
    initial_processed: int = 0,
    initial_words: list[str] | None = None,
    initial_phrases: list[str] | None = None,
    initial_sentences: list[str] | None = None,
    initial_generation_seconds: float = 0.0,
    initial_analysis_seconds: float = 0.0,
    initial_carry_text: str = "",
    on_batch_complete: Callable[[dict[str, object]], None] | None = None,
) -> tuple[list[str], list[str], list[str], float, float]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    if analysis_overlap < 0:
        raise ValueError("analysis_overlap cannot be negative")
    if not 0 <= initial_processed <= character_count:
        raise ValueError("initial_processed must be within 0..character_count")

    unique_words: dict[str, None] = dict.fromkeys(initial_words or [])
    unique_phrases: dict[str, None] = dict.fromkeys(initial_phrases or [])
    unique_sentences: dict[str, None] = dict.fromkeys(initial_sentences or [])
    generation_seconds = initial_generation_seconds
    analysis_seconds = initial_analysis_seconds
    processed = initial_processed
    carry_text = initial_carry_text

    while processed < character_count:
        current_batch_size = min(batch_size, character_count - processed)

        generation_start = time.perf_counter()
        chunk_text = generate_monkey_text(
            current_batch_size,
            character_pool=character_pool,
            stream=stream,
            render_output=render_output,
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
            elapsed = generation_seconds + analysis_seconds
            remaining = character_count - processed
            rate = processed / elapsed if elapsed > 0 else 0.0
            eta_seconds = int(remaining / rate) if rate > 0 else -1
            eta_label = "unknown"
            if eta_seconds >= 0:
                eta_minutes, eta_secs = divmod(eta_seconds, 60)
                eta_hours, eta_minutes = divmod(eta_minutes, 60)
                eta_label = f"{eta_hours:02d}:{eta_minutes:02d}:{eta_secs:02d}"

            print()
            print(
                f"Processed {processed:,}/{character_count:,} characters "
                f"({processed / character_count:.1%}) | ETA {eta_label}"
            )

        if on_batch_complete is not None:
            on_batch_complete(
                {
                    "processed": processed,
                    "words": list(unique_words),
                    "phrases": list(unique_phrases),
                    "sentences": list(unique_sentences),
                    "generation_seconds": generation_seconds,
                    "analysis_seconds": analysis_seconds,
                    "carry_text": carry_text,
                }
            )

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
    run_mode = prompt_run_mode()
    character_pool = get_character_pool(character_mode)
    batch_size = 10_000
    output_stream: TextIO | None = None
    render_output = run_mode != 2
    output_path: Path | None = None
    checkpoint_enabled = prompt_yes_no("Enable checkpoint/resume for this run?", default_yes=True)
    checkpoint_path: Path | None = None
    checkpoint_data: dict[str, object] | None = None
    initial_processed = 0
    initial_words: list[str] = []
    initial_phrases: list[str] = []
    initial_sentences: list[str] = []
    initial_generation_seconds = 0.0
    initial_analysis_seconds = 0.0
    initial_carry_text = ""
    rng = random.Random()
    batches_since_checkpoint = 0
    checkpoint_save_failures = 0

    if checkpoint_enabled:
        checkpoint_path = prompt_checkpoint_path()
        checkpoint_data = load_checkpoint(checkpoint_path)

    if run_mode == 3:
        output_path = prompt_output_file_path()
        output_path.parent.mkdir(parents=True, exist_ok=True)

    if checkpoint_data is not None and checkpoint_path is not None:
        checkpoint_matches = (
            checkpoint_data.get("character_count") == character_count
            and checkpoint_data.get("character_mode") == character_mode
            and checkpoint_data.get("run_mode") == run_mode
            and checkpoint_data.get("output_path") == (str(output_path) if output_path else None)
            and checkpoint_data.get("completed") is False
        )

        if checkpoint_matches and prompt_yes_no(
            f"Resume from checkpoint at {checkpoint_path}?", default_yes=True
        ):
            initial_processed = int(checkpoint_data.get("processed", 0))
            initial_words = list(checkpoint_data.get("words", []))
            initial_phrases = list(checkpoint_data.get("phrases", []))
            initial_sentences = list(checkpoint_data.get("sentences", []))
            initial_generation_seconds = float(checkpoint_data.get("generation_seconds", 0.0))
            initial_analysis_seconds = float(checkpoint_data.get("analysis_seconds", 0.0))
            initial_carry_text = str(checkpoint_data.get("carry_text", ""))

            encoded_random_state = checkpoint_data.get("random_state")
            if isinstance(encoded_random_state, str):
                rng.setstate(_deserialize_random_state(encoded_random_state))

            print(f"Resuming from {initial_processed:,}/{character_count:,} characters.")
        elif checkpoint_matches:
            print("Checkpoint found but starting a fresh run by request.")
        else:
            print("Checkpoint exists but does not match this run; starting a fresh run.")

    if run_mode == 3:
        if output_path is None:
            raise RuntimeError("Output path is required for file output mode")

        file_mode = "a" if initial_processed > 0 else "w"
        output_stream = output_path.open(file_mode, encoding="utf-8", newline="")

    def checkpoint_callback(batch_state: dict[str, object]) -> None:
        nonlocal batches_since_checkpoint
        nonlocal checkpoint_save_failures
        if not checkpoint_enabled or checkpoint_path is None:
            return

        batches_since_checkpoint += 1
        processed = int(batch_state["processed"])
        is_final_batch = processed >= character_count
        if not is_final_batch and batches_since_checkpoint < CHECKPOINT_BATCH_INTERVAL:
            return

        batches_since_checkpoint = 0
        try:
            save_checkpoint(
                checkpoint_path,
                {
                    "version": CHECKPOINT_VERSION,
                    "character_count": character_count,
                    "character_mode": character_mode,
                    "run_mode": run_mode,
                    "output_path": str(output_path) if output_path else None,
                    "processed": processed,
                    "words": batch_state["words"],
                    "phrases": batch_state["phrases"],
                    "sentences": batch_state["sentences"],
                    "generation_seconds": batch_state["generation_seconds"],
                    "analysis_seconds": batch_state["analysis_seconds"],
                    "carry_text": batch_state["carry_text"],
                    "random_state": _serialize_random_state(rng.getstate()),
                    "completed": is_final_batch,
                },
            )
        except OSError as error:
            checkpoint_save_failures += 1
            print(
                "Warning: checkpoint save failed; continuing run without interruption. "
                f"({error})"
            )

    print()
    print(f"Typing {character_count} characters...")
    print(f"Running in batches of {batch_size:,} characters.")
    if run_mode == 2:
        print("Fast mode enabled: character rendering is disabled.")
    if run_mode == 3 and output_path is not None:
        print(f"Fast mode enabled: writing generated text to {output_path}.")
    print()

    try:
        words, phrases, sentences, generation_seconds, analysis_seconds = run_batched_simulation(
            character_count=character_count,
            character_pool=character_pool,
            batch_size=batch_size,
            stream=output_stream,
            render_output=render_output,
            random_choice=rng.choice,
            initial_processed=initial_processed,
            initial_words=initial_words,
            initial_phrases=initial_phrases,
            initial_sentences=initial_sentences,
            initial_generation_seconds=initial_generation_seconds,
            initial_analysis_seconds=initial_analysis_seconds,
            initial_carry_text=initial_carry_text,
            on_batch_complete=checkpoint_callback if checkpoint_enabled else None,
        )
    finally:
        if output_stream is not None:
            output_stream.close()

    print(f"Analysis complete in {analysis_seconds:.2f}s (generation: {generation_seconds:.2f}s).")
    display_results(words, phrases, sentences)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())