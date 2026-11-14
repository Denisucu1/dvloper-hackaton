"""Offline, deterministic tests for the three resilience fixes ported from origin/main.

- llm.Gateway: one short retry on a busy gateway (429 / 5xx), inside the caller's timeout;
- prep.run_prep: an incomplete cache is rebuilt when the gateway can afford it;
- main._ready_timeout: `ready` gets what is left of the study clock after a cold start.

No network, no sleeps, no keys.
"""
import contextlib
import io
import json
import tempfile
import unittest
import urllib.error
from unittest import mock

from harness import config, memory, prep
from harness.game_model import ARTIFACT_VERSION
from harness.llm import Gateway

RULES = """BOARD RUSH - 5x5. Two players alternate placing a mark on an empty cell.
Get four in a row to win. A move is a cell name like C3."""

SAMPLE = {
    "observation": "BOARD RUSH. You are seat 0. Empty board.\nIt is your turn.",
    "legal_actions": ["C3"],
    "seat": 0,
    "turn": 1,
}

SPEC = {
    "family": "think_ahead",
    "action_format": "cell like C3",
    "action_examples": ["C3"],
    "winning_idea": "four in a row",
    "hidden_information": False,
    "simultaneous": False,
    "scoring": "line of four wins",
    "speech_rules": "",
    "forbidden_subjects": [],
    "traps": [],
    "notes": "5x5",
    "playbook": ["Take the centre."],
}


# --------------------------------------------------------------------------- gateway retry
class FakeResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def completion(text: str) -> dict:
    return {
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://gw.test/v1/chat/completions", code, "boom", None, None)


def gateway() -> Gateway:
    client = Gateway()
    client.configure({"base_url": "http://gw.test/v1", "api_key": "sk-test", "model": "haiku"}, cap=0.75)
    return client


class GatewayRetryTests(unittest.TestCase):
    def ask(self, side_effect, timeout):
        # HTTPError carries a file, so close the scripted ones instead of leaving them to the GC.
        errors = [item for item in side_effect if isinstance(item, urllib.error.HTTPError)]
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                with mock.patch("urllib.request.urlopen", side_effect=side_effect) as urlopen, \
                        mock.patch("harness.llm.time.sleep") as sleep:
                    client = gateway()
                    text = client.ask("system", "user", timeout=timeout)
        finally:
            for error in errors:
                with contextlib.suppress(Exception):
                    error.close()
        return client, text, urlopen, sleep

    def test_429_is_retried_once_within_the_timeout(self):
        client, text, urlopen, sleep = self.ask(
            [http_error(429), FakeResponse(completion("hello"))], timeout=15.0
        )
        self.assertEqual(text, "hello")
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once()
        self.assertEqual(client.calls, 1)
        self.assertEqual(client.failures, 1)
        self.assertGreater(client.spend, 0.0)
        retry_timeout = urlopen.call_args_list[1].kwargs["timeout"]
        self.assertGreater(retry_timeout, 0.0)
        self.assertLessEqual(retry_timeout, 15.0)

    def test_5xx_is_retryable(self):
        for code in (500, 502, 503, 504, 529):
            with self.subTest(code=code):
                client, text, urlopen, _ = self.ask(
                    [http_error(code), FakeResponse(completion("ok"))], timeout=15.0
                )
                self.assertEqual(text, "ok")
                self.assertEqual(urlopen.call_count, 2)
                self.assertEqual(client.failures, 1)

    def test_no_retry_when_the_timeout_is_too_small(self):
        client, text, urlopen, sleep = self.ask([http_error(429), http_error(429)], timeout=1.0)
        self.assertEqual(text, "")
        self.assertEqual(urlopen.call_count, 1)
        sleep.assert_not_called()
        self.assertEqual(client.calls, 0)
        self.assertEqual(client.failures, 1)

    def test_no_retry_on_a_non_retryable_status(self):
        client, text, urlopen, sleep = self.ask(
            [http_error(400), FakeResponse(completion("nope"))], timeout=15.0
        )
        self.assertEqual(text, "")
        self.assertEqual(urlopen.call_count, 1)
        sleep.assert_not_called()
        self.assertEqual(client.calls, 0)
        self.assertEqual(client.failures, 1)


# --------------------------------------------------------------------------- prep cache guard
class FakeGateway:
    """A gateway whose spec calls are scripted; `failures` stays at zero (no transport error)."""

    def __init__(self, replies=None, affordable=True):
        self.replies = list(replies or [])
        self.calls = 0
        self.failures = 0
        self.spend = 0.0
        self.cap = 0.75
        self._affordable = affordable

    def configure(self, info=None, cap=None):
        return None

    @property
    def available(self):
        return True

    @property
    def affordable(self):
        return self._affordable

    def ask_json(self, system, user, **kwargs):
        self.calls += 1
        if not self.replies:
            return None
        return self.replies.pop(0)

    def ask(self, system, user, **kwargs):
        raise AssertionError("code generation should be disabled in these tests")

    def summary(self):
        return {"calls": self.calls, "spend_usd": self.spend, "cap_usd": self.cap,
                "failures": self.failures}


def message() -> dict:
    return {
        "type": "prep", "game_id": "cache_probe", "game_name": "Board Rush",
        "variants": [{"label": "5x5", "rules": RULES, "sample_turn": dict(SAMPLE)}],
        "seconds": 600, "you": "adn", "study_seconds": 30, "move_seconds": 20,
        "llm": {"base_url": "scripted://", "api_key": "sk-test", "model": "haiku"},
    }


class CacheGuardTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="adn-cache-guard-")
        self.patch = mock.patch.object(config, "MEMORY_DIR", self.root)
        self.patch.start()
        self.patches = [
            mock.patch.object(config, "GENPOLICY", False),
            mock.patch.object(prep, "GENMODEL", False),
        ]
        for patch in self.patches:
            patch.start()
        fp = memory.fingerprint(memory.variants_rules(message()["variants"]))
        self.directory = memory.game_dir("cache_probe")
        self.fp = fp
        self.write_cache()

    def tearDown(self):
        self.patch.stop()
        for patch in self.patches:
            patch.stop()

    def write_cache(self, source="prep", genmodel=""):
        doc = {**SPEC, "game_id": "cache_probe", "fingerprint": self.fp,
               "artifact_version": ARTIFACT_VERSION, "source": source}
        memory.atomic_write_json(self.directory / config.ARTIFACT_SPEC, doc)
        memory.atomic_write_text(self.directory / config.ARTIFACT_PLAYBOOK, "- Take the centre.")
        memory.atomic_write_json(self.directory / "manifest.json", {
            "game_id": "cache_probe", "fingerprint": self.fp,
            "artifact_version": ARTIFACT_VERSION, "genmodel": genmodel, "genpolicy": "",
        })
        if genmodel:
            memory.atomic_write_text(self.directory / genmodel, "def choose(*a):\n    return 'C3'\n")

    def run_prep(self, fake):
        with contextlib.redirect_stderr(io.StringIO()):
            return prep.run_prep(message(), fake)

    def test_fallback_source_is_rebuilt(self):
        self.write_cache(source="fallback")
        fake = FakeGateway([dict(SPEC)])
        result = self.run_prep(fake)
        self.assertFalse(result.cached)
        self.assertEqual(result.spec.source, "prep")
        self.assertEqual(fake.calls, 1)
        self.assertIn("spec from model", " | ".join(result.notes))

    def test_missing_simulator_is_rebuilt(self):
        self.write_cache(source="prep", genmodel="")
        fake = FakeGateway([dict(SPEC)])
        result = self.run_prep(fake)
        self.assertFalse(result.cached)
        self.assertEqual(fake.calls, 1)

    def test_whole_cache_is_served_without_a_call(self):
        self.write_cache(source="prep", genmodel="probe-model.py")
        fake = FakeGateway([])
        result = self.run_prep(fake)
        self.assertTrue(result.cached)
        self.assertEqual(result.spec.source, "prep")
        self.assertTrue(result.genmodel)
        self.assertEqual(fake.calls, 0)
        self.assertIn("cached artifact reused", result.notes)

    def test_degraded_cache_is_reused_when_there_is_no_budget(self):
        self.write_cache(source="fallback")
        fake = FakeGateway([], affordable=False)
        result = self.run_prep(fake)
        self.assertTrue(result.cached)
        self.assertEqual(fake.calls, 0)


class SpecWaitTests(unittest.TestCase):
    """The growing waits only follow a gateway that actually refused the calls."""

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="adn-spec-wait-")
        self.memory_patch = mock.patch.object(config, "MEMORY_DIR", self.root)
        self.memory_patch.start()
        self.patches = [
            mock.patch.object(config, "GENPOLICY", False),
            mock.patch.object(prep, "GENMODEL", False),
            mock.patch.object(config, "PREP_SPEC_WAITS", (0.01,)),
        ]
        for patch in self.patches:
            patch.start()

    def tearDown(self):
        for patch in self.patches:
            patch.stop()
        self.memory_patch.stop()

    def test_a_refused_gateway_is_retried_after_a_wait(self):
        fake = FakeGateway([])

        def ask_json(system, user, **kwargs):
            fake.calls += 1
            fake.failures += 1
            return None

        fake.ask_json = ask_json
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with mock.patch("harness.prep.time.sleep") as sleep:
                result = prep.run_prep(message(), fake)
        self.assertEqual(fake.calls, 3)  # first, immediate retry, then one waited retry
        sleep.assert_called_once()
        self.assertIn("waiting to try again", stderr.getvalue())
        self.assertIn("regex fallback spec", " | ".join(result.notes))

    def test_an_unparseable_answer_falls_back_without_waiting(self):
        fake = FakeGateway([None, None])
        with contextlib.redirect_stderr(io.StringIO()):
            with mock.patch("harness.prep.time.sleep") as sleep:
                result = prep.run_prep(message(), fake)
        self.assertEqual(fake.calls, 2)
        sleep.assert_not_called()
        self.assertIn("regex fallback spec", " | ".join(result.notes))


# --------------------------------------------------------------------------- ready clock
class ReadyClockTests(unittest.TestCase):
    def test_ready_timeout_is_positive_and_shrinks_with_elapsed(self):
        from main import _ready_timeout

        fresh = _ready_timeout(30, 0.0)
        later = _ready_timeout(30, 10.0)
        almost_done = _ready_timeout(30, 20.0)
        self.assertGreater(fresh, 0.0)
        self.assertGreater(fresh, later)
        self.assertGreater(later, almost_done)
        self.assertAlmostEqual(fresh, 30.0 - config.SAFETY, places=3)

    def test_ready_timeout_never_goes_non_positive(self):
        from main import _ready_timeout

        self.assertGreater(_ready_timeout(10, 999.0), 0.0)
        self.assertGreater(_ready_timeout(0.0, 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
