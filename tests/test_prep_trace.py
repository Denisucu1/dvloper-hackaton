import contextlib
import importlib.util
import io
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEC = importlib.util.spec_from_file_location("prep_trace", os.path.join(ROOT, "scripts", "prep_trace.py"))
prep_trace = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(prep_trace)


class OfflineTraceTests(unittest.TestCase):
    """The offline trace is deterministic: no gateway, no key, no network."""

    @classmethod
    def setUpClass(cls):
        with contextlib.redirect_stdout(io.StringIO()):
            cls.result = prep_trace.run_trace(real=False)

    def test_specialist_is_built(self):
        self.assertTrue(self.result["ok"])
        self.assertEqual(self.result["validation"], {"genpolicy": True, "genmodel": True})
        self.assertTrue(self.result["manifest"]["family"])

    def test_all_prep_stages_ran(self):
        details = [event["detail"] for event in self.result["events"]]
        joined = " | ".join(details)
        self.assertIn("spec extraction", joined)
        self.assertIn("policy generation", joined)
        self.assertIn("simulator generation", joined)
        self.assertIn("prep finished", joined)

    def test_validation_covered_both_variants(self):
        notes = " | ".join(self.result["manifest"]["notes"])
        self.assertIn("ok on 2 sample(s)", notes)

    def test_artifacts_are_the_expected_set(self):
        names = set(self.result["artifacts"])
        for required in ("spec.json", "manifest.json", "playbook.txt", "rules.txt", "samples.json"):
            self.assertIn(required, names)
        self.assertTrue(any(name.endswith("-model.py") for name in names))
        self.assertTrue(any(name.endswith("-policy.py") for name in names))

    def test_prep_flows_into_a_move(self):
        self.assertEqual(self.result["engine_action"], "pick 5")
        sources = [event["detail"] for event in self.result["events"] if event["stage"] == "engine"]
        self.assertTrue(sources and "generated model" in sources[0])

    def test_three_model_calls_and_no_network(self):
        # spec, policy, simulator - plus, since the prep clock is used for more than a few seconds, the
        # strategy brief and one chat plan per variant (the offline gateway answers those with nothing).
        self.assertGreaterEqual(self.result["calls"], 3)
        self.assertLessEqual(self.result["calls"], 3 + 1 + 4)
        for event in self.result["events"]:
            self.assertNotIn("http", event["detail"].lower())

    def test_memory_dir_is_cleaned_by_default(self):
        self.assertIsNone(self.result["memory_dir"])
        self.assertFalse(os.path.exists("/tmp/never-created"))


class SyntheticInputTests(unittest.TestCase):
    def test_message_has_two_variants_with_their_own_samples(self):
        msg = prep_trace.synthetic_message()
        self.assertEqual(len(msg["variants"]), 2)
        labels = [variant["label"] for variant in msg["variants"]]
        self.assertEqual(labels, ["3 numbers", "5 numbers"])
        for variant in msg["variants"]:
            legal = variant["sample_turn"]["legal_actions"]
            self.assertEqual(len(legal), variant["size"])

    def test_scripted_gateway_needs_no_key(self):
        gateway = prep_trace.ScriptedGateway()
        self.assertTrue(gateway.available)
        self.assertTrue(gateway.affordable)


if __name__ == "__main__":
    unittest.main()
