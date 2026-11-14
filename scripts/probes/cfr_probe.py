"""Probe: tabular CFR+ (``harness/cfr.py``) on Kuhn poker.

Small deterministic solve (500 CFR+ iterations, not the 20k offline default):

- the average policy covers all 12 Kuhn information sets;
- every information-set table is a probability distribution (non-negative, sums to 1);
- the exact best-response exploitability proxy is below a loose 0.05 threshold;
- two identical solves are bit-for-bit deterministic.

Score is the fraction of boolean checks that pass. The exploitability threshold is deliberately loose:
the probe is a regression alarm, not a convergence benchmark.
"""
from __future__ import annotations

from harness import cfr

from scripts.probes.base import Probe, collect, register, summarize

ITERATIONS = 500
EXPLOITABILITY_LIMIT = 0.05


@register
class CfrProbe(Probe):
    name = "solvers.cfr"
    component = "cfr"

    def run(self) -> dict:
        facts: dict = {}
        policy = cfr.solve_kuhn(iterations=ITERATIONS)
        facts["iterations"] = ITERATIONS
        facts["infosets"] = len(policy.strategies)

        def covers_all_infosets() -> bool:
            expected = set(cfr.kuhn_infosets())
            return set(policy.strategies) == expected and len(policy.strategies) == 12

        def tables_are_distributions() -> bool:
            deviations = []
            negative = []
            for infoset, table in policy.strategies.items():
                total = sum(float(value) for value in table.values())
                deviations.append(abs(total - 1.0))
                if any(float(value) < 0.0 for value in table.values()):
                    negative.append(infoset)
            facts["max_sum_deviation"] = round(max(deviations, default=0.0), 12)
            facts["negative_tables"] = negative[:5]
            return not negative and all(deviation < 1e-9 for deviation in deviations)

        def exploitability_is_low() -> bool:
            value = cfr.kuhn_exploitability(policy)
            facts["exploitability"] = round(value, 6)
            return value < EXPLOITABILITY_LIMIT

        def solve_is_deterministic() -> bool:
            again = cfr.solve_kuhn(iterations=ITERATIONS)
            facts["deterministic"] = again.strategies == policy.strategies
            return again.strategies == policy.strategies

        run = collect([
            ("average policy covers all 12 infosets", covers_all_infosets),
            ("every table sums to 1", tables_are_distributions),
            (f"exploitability proxy below {EXPLOITABILITY_LIMIT}", exploitability_is_low),
            ("identical solves are deterministic", solve_is_deterministic),
        ])
        details = dict(facts)
        details["failures"] = run.failures
        notes = (
            f"{facts['infosets']} infosets, exploitability {facts.get('exploitability')}, "
            f"{ITERATIONS} iterations"
        )
        return summarize(run, notes=notes, details=details)
