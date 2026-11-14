import unittest

from harness import actions


class NormalizeTests(unittest.TestCase):
    def test_exact_match_wins(self):
        self.assertEqual(actions.normalize("B2", ["A1", "B2"]), "B2")

    def test_case_and_quotes(self):
        self.assertEqual(actions.normalize(' "b2" ', ["A1", "B2"]), "B2")

    def test_embedded_action_is_recovered(self):
        self.assertEqual(actions.normalize("I will play C3 now", ["A1", "C3"]), "C3")

    def test_illegal_returns_none(self):
        self.assertIsNone(actions.normalize("Z9", ["A1", "B2"]))

    def test_no_legal_list_keeps_clean_action(self):
        self.assertEqual(actions.normalize(" guess 42 ", None), "guess 42")

    def test_clean_collapses_newlines(self):
        self.assertEqual(actions.clean("hold\n"), "hold")
        self.assertEqual(actions.clean(None), "")


class FallbackTests(unittest.TestCase):
    def test_avoids_repeats(self):
        self.assertEqual(actions.choose_fallback(["A1", "B2"], ["A1"]), "B2")

    def test_everything_seen_falls_back_to_first(self):
        self.assertEqual(actions.choose_fallback(["A1", "B2"], ["A1", "B2"]), "A1")

    def test_no_list_has_no_fallback(self):
        self.assertIsNone(actions.choose_fallback(None))
        self.assertIsNone(actions.choose_fallback([]))


class PlausibleTests(unittest.TestCase):
    def test_rejects_garbage(self):
        for bad in ("", "{'action': 'A1'}", "null", "x" * 200):
            self.assertFalse(actions.is_plausible(bad), bad)

    def test_accepts_format_examples(self):
        self.assertTrue(actions.is_plausible("guess 42", ["guess <n>"]))
        self.assertTrue(actions.is_plausible("A1", ["A1"]))


if __name__ == "__main__":
    unittest.main()
