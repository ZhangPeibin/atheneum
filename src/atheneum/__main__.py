"""Entry point for ``python -m atheneum``.

Runs the retrieval evaluation, which is the one thing worth doing without a CLI
installed — it verifies the index end to end on a labelled dataset.
"""

from __future__ import annotations

import json
import sys

from atheneum.evaluate import run_evaluation


def main(argv: list[str]) -> int:
    if any(arg in {"-h", "--help"} for arg in argv):
        print("usage: python -m atheneum [--json]\n\nScores retrieval on the bundled dataset.")
        return 0
    report = run_evaluation()
    print(json.dumps(report, indent=2, sort_keys=True))
    for row in report["results"]:
        print(
            f"{row['name']:<8} recall@{report['k']}={row['recall_at_k']:.3f} "
            f"mrr={row['mrr']:.3f} hit={row['hit_rate']:.3f} "
            f"latency={row['mean_latency_ms']:.3f}ms",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
