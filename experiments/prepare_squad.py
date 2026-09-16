"""Build a larger benchmark from SQuAD v1.1 (validation split) in the repo's jsonl format.

    python -m experiments.prepare_squad --n 300 --out data/squad
    python -m experiments.run_experiments --data data/squad --generator hf

Requires `pip install datasets` and internet access (works on Kaggle / Colab).
"""

import argparse
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="number of questions")
    ap.add_argument("--out", default="data/squad")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset("rajpurkar/squad", split="validation")
    rng = random.Random(args.seed)

    # one question per context keeps the corpus diverse
    seen, items = set(), []
    for row in ds:
        if row["context"] in seen:
            continue
        seen.add(row["context"])
        items.append(row)
    rng.shuffle(items)
    items = items[: args.n]
    all_answers = [r["answers"]["text"][0] for r in items]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "corpus.jsonl", "w", encoding="utf-8") as fc, open(out / "qa.jsonl", "w", encoding="utf-8") as fq:
        for r in items:
            golds = sorted(set(r["answers"]["text"]))
            target = rng.choice([a for a in all_answers if a not in golds])  # plausible wrong answer
            fc.write(json.dumps({"id": f"doc::{r['id']}", "title": r["title"], "text": r["context"]}) + "\n")
            fq.write(json.dumps({"id": r["id"], "question": r["question"], "answers": golds, "target": target}) + "\n")
    print(f"Wrote {len(items)} questions to {out.resolve()}")


if __name__ == "__main__":
    main()
