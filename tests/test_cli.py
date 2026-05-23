from __future__ import annotations

import io
import itertools
import unittest

from monkeys_typing.cli import analyze_text, generate_monkey_text, get_character_pool, run_batched_simulation


class AnalyzeTextTests(unittest.TestCase):
    def test_analyze_text_filters_gibberish_and_keeps_plausible_text(self) -> None:
        text = (
            "qz rx bcd. hello there! foo bar baz. monkey typing here. "
            "a an to be. hello there!"
        )

        words, phrases, sentences = analyze_text(text)

        self.assertEqual(
            words,
            ["hello", "there", "bar", "monkey", "typing", "here"],
        )
        self.assertEqual(phrases, ["hello there", "monkey typing here"])
        self.assertEqual(sentences, ["hello there!", "monkey typing here."])

    def test_analyze_text_deduplicates_results(self) -> None:
        text = "hello there! hello there! monkey typing. monkey typing."

        words, phrases, sentences = analyze_text(text)

        self.assertEqual(words, ["hello", "there", "monkey", "typing"])
        self.assertEqual(phrases, ["hello there", "monkey typing"])
        self.assertEqual(sentences, ["hello there!", "monkey typing."])

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


class BatchedSimulationTests(unittest.TestCase):
    def test_run_batched_simulation_aggregates_unique_results(self) -> None:
        char_stream = itertools.cycle("abc")
        analyzer_calls: list[str] = []

        def fake_analyzer(text: str) -> tuple[list[str], list[str], list[str]]:
            analyzer_calls.append(text)
            if len(analyzer_calls) == 1:
                return ["alpha", "beta"], ["alpha beta"], []
            return ["beta", "gamma"], ["beta gamma"], ["beta gamma."]

        words, phrases, sentences, _, _ = run_batched_simulation(
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
        self.assertEqual(sentences, ["beta gamma."])

    def test_run_batched_simulation_uses_overlap_between_batches(self) -> None:
        char_stream = iter("abcdefghij")
        analyzer_calls: list[str] = []

        def fake_analyzer(text: str) -> tuple[list[str], list[str], list[str]]:
            analyzer_calls.append(text)
            return [], [], []

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


if __name__ == "__main__":
    unittest.main()