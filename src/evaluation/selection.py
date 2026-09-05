"""
selection.py
============
Configuration selection for the phase-based experimental protocol.

Every phase compares configurations on the SAME development posts, so the
per-post outcomes are PAIRED binary data. Independent per-configuration
confidence intervals ignore that pairing and are badly under-powered:
two configurations can differ significantly while their separate CIs overlap
almost completely. This module therefore uses paired tests throughout:

  - k = 2 configurations : McNemar's exact test (binomial on discordant pairs).
                           The exact form is used rather than the chi-square
                           approximation because the discordant count at n = 30
                           is small.
  - k >= 3 configurations: Cochran's Q omnibus test, followed, only if Q is
                           significant, by pairwise McNemar with Holm-Bonferroni
                           correction.
  - effect size          : paired bootstrap CI on the DIFFERENCE between two
                           configurations (resampling posts, not configurations),
                           which respects the pairing that a per-configuration CI
                           discards.

The same machinery is applied to any paired binary outcome, so accuracy and
the grounding criteria (C1/C2/C3) are tested identically.

`select_winner()` implements the pre-specified decision rule described in the
methodology: statistical evidence on accuracy first, then grounding (C3), then
efficiency, then parsimony. The rule is deterministic and applied identically
in every phase, so no phase outcome depends on post-hoc judgement.

No SciPy dependency: the chi-square upper tail is computed from a standard
regularised incomplete gamma implementation, validated in the self-test at the
bottom of this file (`python -m src.evaluation.selection`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Sequence

import numpy as np

ALPHA = 0.05

__all__ = [
    "mcnemar_exact", "cochrans_q", "paired_bootstrap_diff", "holm_correction",
    "PhaseOutcome", "select_winner", "format_phase_report",
]


# ── Chi-square upper tail (no SciPy) ──────────────────────────────────────────
def _gammap_series(a: float, x: float) -> float:
    """Regularised lower incomplete gamma P(a,x) by series expansion."""
    ap, total, term = a, 1.0 / a, 1.0 / a
    for _ in range(1000):
        ap += 1.0
        term *= x / ap
        total += term
        if abs(term) < abs(total) * 1e-15:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gammaq_cf(a: float, x: float) -> float:
    """Regularised upper incomplete gamma Q(a,x) by continued fraction."""
    tiny = 1e-300
    b, c, d = x + 1.0 - a, 1.0 / tiny, 1.0 / (x + 1.0 - a)
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h * math.exp(-x + a * math.log(x) - math.lgamma(a))


def chi2_sf(x: float, df: int) -> float:
    """Upper-tail probability of the chi-square distribution."""
    if x <= 0:
        return 1.0
    a = df / 2.0
    return 1.0 - _gammap_series(a, x / 2.0) if x < a + 1.0 else _gammaq_cf(a, x / 2.0)


# ── Paired tests ──────────────────────────────────────────────────────────────
def mcnemar_exact(a: Sequence[int], b: Sequence[int]) -> dict:
    """
    McNemar's exact test on two paired binary outcome vectors.

    Only discordant pairs carry information: n01 (a wrong, b right) and n10
    (a right, b wrong). Under H0 each discordant pair is a fair coin flip, so
    the two-sided p-value is the exact binomial tail.
    """
    a, b = np.asarray(a, int), np.asarray(b, int)
    if a.shape != b.shape:
        raise ValueError("paired vectors must have equal length")
    n01 = int(np.sum((a == 0) & (b == 1)))
    n10 = int(np.sum((a == 1) & (b == 0)))
    n = n01 + n10
    if n == 0:
        p = 1.0
    else:
        k = min(n01, n10)
        tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
        p = min(1.0, 2.0 * tail)
    return {"n_discordant": n, "n01": n01, "n10": n10, "p_value": p,
            "significant": p < ALPHA}


def cochrans_q(outcomes: dict[str, Sequence[int]]) -> dict:
    """
    Cochran's Q omnibus test for k >= 3 paired binary outcome vectors.

    H0: all configurations have the same success probability. Q is
    chi-square distributed with k-1 degrees of freedom under H0.
    """
    names = list(outcomes)
    mat = np.array([np.asarray(outcomes[n], int) for n in names])  # k x n
    k, n = mat.shape
    if k < 3:
        raise ValueError("Cochran's Q requires at least 3 configurations")
    col = mat.sum(axis=0)                # successes per post across configs
    row = mat.sum(axis=1)                # successes per config
    denom = (k * col.sum() - int((col ** 2).sum()))
    if denom == 0:                       # all configs identical on every post
        return {"Q": 0.0, "df": k - 1, "p_value": 1.0, "significant": False}
    q = (k - 1) * (k * int((row ** 2).sum()) - int(row.sum()) ** 2) / denom
    p = chi2_sf(q, k - 1)
    return {"Q": float(q), "df": k - 1, "p_value": float(p),
            "significant": p < ALPHA}


def holm_correction(pvals: dict[tuple[str, str], float]) -> dict[tuple[str, str], dict]:
    """Holm-Bonferroni step-down correction over a family of pairwise tests."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, prev = {}, 0.0
    for i, (pair, p) in enumerate(items):
        adj = max(prev, min(1.0, (m - i) * p))
        prev = adj
        out[pair] = {"p_raw": p, "p_holm": adj, "significant": adj < ALPHA}
    return out


def paired_bootstrap_diff(a: Sequence[int], b: Sequence[int],
                          n_boot: int = 10_000, seed: int = 42) -> dict:
    """
    Bootstrap CI for the DIFFERENCE in success rate (a - b), resampling posts.

    Posts are resampled once per replicate and both configurations are scored
    on the same resampled posts, preserving the pairing.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float(a.mean() - b.mean()), "ci_lo": float(lo),
            "ci_hi": float(hi), "excludes_zero": bool(lo > 0 or hi < 0)}


# ── Decision rule ─────────────────────────────────────────────────────────────
@dataclass
class PhaseOutcome:
    """Result of applying the selection rule to one phase."""
    phase: str
    winner: str
    step: int
    reason: str
    accuracy: dict[str, float] = field(default_factory=dict)
    grounding_c3: dict[str, float] = field(default_factory=dict)
    latency_s: dict[str, float] = field(default_factory=dict)
    omnibus: dict | None = None
    pairwise: dict | None = None
    effect: dict | None = None


def _omnibus(outcomes: dict[str, Sequence[int]]) -> dict:
    names = list(outcomes)
    if len(names) == 2:
        res = mcnemar_exact(outcomes[names[0]], outcomes[names[1]])
        res["test"] = "McNemar exact"
        return res
    res = cochrans_q(outcomes)
    res["test"] = "Cochran's Q"
    return res


def select_winner(
    phase: str,
    correct: dict[str, Sequence[int]],
    grounding_c3: dict[str, Sequence[int]] | None = None,
    latency_s: dict[str, float] | None = None,
    complexity_rank: dict[str, int] | None = None,
) -> PhaseOutcome:
    """
    Apply the pre-specified four-step selection rule to one phase.

    Step 1 — accuracy, if the paired omnibus test is significant: take the
             highest-accuracy configuration.
    Step 2 — otherwise the configurations are statistically indistinguishable
             on accuracy, so decide on grounding C3 (content relevance), tested
             the same paired way. Highest C3 wins if that test is significant.
    Step 3 — otherwise decide on efficiency: lowest mean latency.
    Step 4 — otherwise prefer the least complex configuration (parsimony).

    Steps 3 and 4 are tie-breaks among options for which no evidence of a
    difference was found, and are recorded as such rather than presented as
    demonstrated superiority.
    """
    names = list(correct)
    acc = {n: float(np.mean(correct[n])) for n in names}
    out = PhaseOutcome(phase=phase, winner="", step=0, reason="", accuracy=acc)

    # Step 1 — accuracy
    omni = _omnibus(correct)
    out.omnibus = omni
    if len(names) > 2 and omni["significant"]:
        raw = {(a, b): mcnemar_exact(correct[a], correct[b])["p_value"]
               for a, b in combinations(names, 2)}
        out.pairwise = holm_correction(raw)
    if omni["significant"]:
        best = max(names, key=lambda n: acc[n])
        runner = max((n for n in names if n != best), key=lambda n: acc[n])
        out.effect = paired_bootstrap_diff(correct[best], correct[runner])
        out.winner, out.step = best, 1
        out.reason = (f"accuracy difference significant "
                      f"({omni['test']}, p = {omni['p_value']:.4f}); "
                      f"highest accuracy selected")
        return out

    # Step 2 — grounding C3
    if grounding_c3:
        out.grounding_c3 = {n: float(np.mean(grounding_c3[n])) for n in names}
        g_omni = _omnibus(grounding_c3)
        if g_omni["significant"]:
            best = max(names, key=lambda n: out.grounding_c3[n])
            runner = max((n for n in names if n != best),
                         key=lambda n: out.grounding_c3[n])
            out.effect = paired_bootstrap_diff(grounding_c3[best],
                                               grounding_c3[runner])
            out.winner, out.step = best, 2
            out.reason = (f"accuracy indistinguishable "
                          f"(p = {omni['p_value']:.4f}); grounding C3 "
                          f"difference significant ({g_omni['test']}, "
                          f"p = {g_omni['p_value']:.4f})")
            return out

    # Step 3 — efficiency
    if latency_s:
        out.latency_s = dict(latency_s)
        best = min(names, key=lambda n: latency_s[n])
        out.winner, out.step = best, 3
        out.reason = ("no significant difference in accuracy or grounding; "
                      "lowest mean latency selected as tie-break")
        return out

    # Step 4 — parsimony
    ranks = complexity_rank or {n: i for i, n in enumerate(names)}
    best = min(names, key=lambda n: ranks[n])
    out.winner, out.step = best, 4
    out.reason = ("no significant difference in accuracy, grounding or "
                  "efficiency; simplest configuration selected as tie-break")
    return out


def format_phase_report(o: PhaseOutcome) -> str:
    """Human-readable phase summary for the notebook and the write-up."""
    lines = [f"{o.phase}", "-" * len(o.phase)]
    lines.append("  accuracy:  " + "  ".join(
        f"{n}={v:.3f}" for n, v in sorted(o.accuracy.items(), key=lambda x: -x[1])))
    if o.grounding_c3:
        lines.append("  C3:        " + "  ".join(
            f"{n}={v:.3f}" for n, v in sorted(o.grounding_c3.items(), key=lambda x: -x[1])))
    if o.latency_s:
        lines.append("  latency:   " + "  ".join(
            f"{n}={v:.1f}s" for n, v in sorted(o.latency_s.items(), key=lambda x: x[1])))
    if o.omnibus:
        lines.append(f"  omnibus:   {o.omnibus['test']}, "
                     f"p = {o.omnibus['p_value']:.4f}"
                     + (f", Q = {o.omnibus['Q']:.3f} (df {o.omnibus['df']})"
                        if "Q" in o.omnibus else
                        f", discordant = {o.omnibus['n_discordant']}"))
    if o.pairwise:
        for (a, b), r in o.pairwise.items():
            lines.append(f"    {a} vs {b}: p = {r['p_raw']:.4f}, "
                         f"Holm p = {r['p_holm']:.4f}")
    if o.effect:
        lines.append(f"  effect:    diff = {o.effect['diff']:+.3f} "
                     f"[{o.effect['ci_lo']:+.3f}, {o.effect['ci_hi']:+.3f}] "
                     f"(paired bootstrap)")
    lines.append(f"  WINNER:    {o.winner}  (rule step {o.step} — {o.reason})")
    return "\n".join(lines)


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # chi2_sf against published chi-square table values
    for x, df, expected in [(3.841, 1, 0.05), (5.991, 2, 0.05),
                            (7.815, 3, 0.05), (11.345, 3, 0.01)]:
        got = chi2_sf(x, df)
        assert abs(got - expected) < 5e-4, f"chi2_sf({x},{df})={got}"
    print("chi2_sf matches chi-square table values")

    # McNemar exact against a hand-computable case: 0 vs 6 discordant
    r = mcnemar_exact([1] * 6 + [1] * 24, [0] * 6 + [1] * 24)
    assert abs(r["p_value"] - 2 * (1 / 64)) < 1e-12, r
    print(f"McNemar exact (6-0 discordant): p = {r['p_value']:.5f}")

    # Pairing matters: identical marginal accuracies, opposite CI verdicts
    rng = np.random.default_rng(0)
    base = rng.integers(0, 2, 30)
    a = base.copy(); b = base.copy()
    a[:5], b[:5] = 1, 0                      # 5 discordant pairs, all one way
    print(f"\nA acc = {a.mean():.3f}, B acc = {b.mean():.3f}")
    print(f"  McNemar exact p = {mcnemar_exact(a, b)['p_value']:.4f}")
    print(f"  paired bootstrap diff CI = "
          f"[{paired_bootstrap_diff(a, b)['ci_lo']:+.3f}, "
          f"{paired_bootstrap_diff(a, b)['ci_hi']:+.3f}]")

    # Full rule on a synthetic three-config phase
    outcomes = {
        "BM25":   list(rng.integers(0, 2, 30)),
        "Dense":  list(rng.integers(0, 2, 30)),
        "Hybrid": list(rng.integers(0, 2, 30)),
    }
    c3 = {"BM25": [1] * 14 + [0] * 16, "Dense": [1] * 17 + [0] * 13,
          "Hybrid": [1] * 15 + [0] * 15}
    lat = {"BM25": 10.4, "Dense": 16.0, "Hybrid": 17.7}
    print()
    print(format_phase_report(
        select_winner("Phase 1 (synthetic)", outcomes, c3, lat)))
