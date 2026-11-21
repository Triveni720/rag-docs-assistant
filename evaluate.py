"""Retrieval evaluation against a golden dataset.

Reports hit rate @k and mean reciprocal rank (MRR): how often the correct
document section is retrieved, and how high it is ranked.
"""
import json
from pathlib import Path

from app.rag import Retriever, load_chunks

GOLDEN = Path(__file__).resolve().parent / "tests" / "golden.json"


def evaluate(k: int = 3) -> dict:
    retriever = Retriever(load_chunks())
    cases = json.loads(GOLDEN.read_text())
    hits, reciprocal_ranks = 0, []
    for case in cases:
        results = retriever.search(case["question"], k=k)
        ranks = [
            i + 1 for i, (c, _) in enumerate(results)
            if c.source == case["source"] and c.section == case["section"]
        ]
        if ranks:
            hits += 1
            reciprocal_ranks.append(1 / ranks[0])
        else:
            reciprocal_ranks.append(0.0)
    return {
        "cases": len(cases),
        f"hit_rate@{k}": round(hits / len(cases), 3),
        "mrr": round(sum(reciprocal_ranks) / len(cases), 3),
    }


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
