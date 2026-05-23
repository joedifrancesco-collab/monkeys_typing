from __future__ import annotations

import io
import itertools
import unittest
from unittest.mock import patch

from monkeys_typing.cli import (
    MAX_CHARACTER_COUNT,
    MIN_CHARACTER_COUNT,
    analyze_text,
    generate_monkey_text,
    get_character_pool,
    prompt_character_count,
    run_batched_simulation,
)


class AnalyzeTextTests(unittest.TestCase):
    def test_analyze_text_filters_gibberish_and_keeps_plausible_text(self) -> None:
        text = (
            "qz rx bcd. hello there! foo bar baz. monkey typing here. "
            "a an to be. hello there!"
        )

        words, phrases = analyze_text(text)

        self.assertEqual(
            words,
            ["hello", "there", "bar", "monkey", "typing", "here"],
        )
        self.assertEqual(phrases, ["hello there", "monkey typing here"])

    def test_analyze_text_deduplicates_results(self) -> None:
        text = "hello there! hello there! monkey typing. monkey typing."

        words, phrases = analyze_text(text)

        self.assertEqual(words, ["hello", "there", "monkey", "typing"])
        self.assertEqual(phrases, ["hello there", "monkey typing"])

    def test_get_character_pool(self) -> None:
        import string

        self.assertEqual(get_character_pool(1), "abcdefghijklmnopqrstuvwxyz ")
        self.assertEqual(get_character_pool(2), "ABCDEFGHIJKLMNOPQRSTUVWXYZ ")
        self.assertEqual(get_character_pool(3), "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ")
        self.assertEqual(get_character_pool(4), string.ascii_letters + " " + string.punctuation)


class GenerateMonkeyTextTests(unittest.TestCase):
    def test_generate_monkey_text_writes_one_character_at_a_time(self) -> None:
        characters = iter("abc")
        stream = io.StringIO()

        text = generate_monkey_text(
            3,
            character_pool="abc",
            stream=stream,
            newline_interval=0,
            typing_delay=0.0,
            random_choice=lambda pool: next(characters),
        )

        self.assertEqual(text, "abc")
        self.assertEqual(stream.getvalue(), "abc")

    def test_generate_monkey_text_can_skip_rendering(self) -> None:
        characters = iter("abc")

        text = generate_monkey_text(
            3,
            character_pool="abc",
            render_output=False,
            newline_interval=0,
            typing_delay=0.0,
            random_choice=lambda pool: next(characters),
        )

        self.assertEqual(text, "abc")


class BatchedSimulationTests(unittest.TestCase):
    def test_run_batched_simulation_aggregates_unique_results(self) -> None:
        char_stream = itertools.cycle("abc")
        analyzer_calls: list[str] = []

        def fake_analyzer(text: str) -> tuple[list[str], list[str]]:
            analyzer_calls.append(text)
            if len(analyzer_calls) == 1:
                return ["alpha", "beta"], ["alpha beta"]
            return ["beta", "gamma"], ["beta gamma"]

        words, phrases, _, _ = run_batched_simulation(
            character_count=15,
            character_pool="abc",
            batch_size=10,
            stream=io.StringIO(),
            newline_interval=0,
            flush_interval=0,
            random_choice=lambda pool: next(char_stream),
            analyzer=fake_analyzer,
            show_batch_progress=False,
        )

        self.assertEqual(len(analyzer_calls), 2)
        self.assertEqual(words, ["alpha", "beta", "gamma"])
        self.assertEqual(phrases, ["alpha beta", "beta gamma"])

    def test_run_batched_simulation_uses_overlap_between_batches(self) -> None:
        char_stream = iter("abcdefghij")
        analyzer_calls: list[str] = []

        def fake_analyzer(text: str) -> tuple[list[str], list[str]]:
            analyzer_calls.append(text)
            return [], []

        run_batched_simulation(
            character_count=10,
            character_pool="abc",
            batch_size=5,
            stream=io.StringIO(),
            newline_interval=0,
            flush_interval=0,
            random_choice=lambda pool: next(char_stream),
            analyzer=fake_analyzer,
            show_batch_progress=False,
            analysis_overlap=2,
        )

        self.assertEqual(analyzer_calls[0], "abcde")
        self.assertEqual(analyzer_calls[1], "defghij")

    def test_run_batched_simulation_can_resume_from_initial_state(self) -> None:
        char_stream = iter("klmnopqrst")

        words, phrases, generation_seconds, analysis_seconds = run_batched_simulation(
            character_count=10,
            character_pool="abc",
            batch_size=5,
            stream=io.StringIO(),
            newline_interval=0,
            flush_interval=0,
            random_choice=lambda pool: next(char_stream),
            analyzer=lambda text: (["resume"], []),
            show_batch_progress=False,
            initial_processed=5,
            initial_words=["existing"],
            initial_phrases=["existing phrase"],
            initial_generation_seconds=1.0,
            initial_analysis_seconds=2.0,
            initial_carry_text="abc",
        )

        self.assertEqual(words, ["existing", "resume"])
        self.assertEqual(phrases, ["existing phrase"])
        self.assertGreaterEqual(generation_seconds, 1.0)
        self.assertGreaterEqual(analysis_seconds, 2.0)


class PromptTests(unittest.TestCase):
    def test_prompt_character_count_accepts_max_limit(self) -> None:
        with patch("builtins.input", side_effect=[str(MIN_CHARACTER_COUNT - 1), str(MAX_CHARACTER_COUNT)]):
            result = prompt_character_count()

        self.assertEqual(result, MAX_CHARACTER_COUNT)


if __name__ == "__main__":
    unittest.main()