"""Vanilla RAG vs ACRC-Guard under increasing numbers of poisoned passages.

Examples
--------
Offline, no downloads (fast smoke run):
    python -m experiments.run_experiments --embedder tfidf --verifier lexical

Full models (Kaggle / Colab / local GPU):
    python -m experiments.run_experiments --generator hf

Outputs (in --out): per_question.csv, summary.csv, summary.md, attack_success.png, accuracy.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acrc_guard import GuardConfig, GuardedRAG, VanillaRAG  # noqa: E402
from acrc_guard.attacks import is_poison, make_poison  # noqa: E402
from acrc_guard.data import load_dataset  # noqa: E402
from acrc_guard.generator import ABSTAIN  # noqa: E402
from acrc_guard.pipeline import Components  # noqa: E402
from acrc_guard.text import contains_answer  # noqa: E402

METHODS = {"Vanilla RAG": VanillaRAG, "ACRC-Guard": GuardedRAG}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/sample")
    ap.add_argument("--out", default="results")
    ap.add_argument("--poison-counts", type=int, nargs="+", default=[0, 1, 3, 5])
    ap.add_argument("--embedder", default=GuardConfig.embedder)
    ap.add_argument("--verifier", default=GuardConfig.verifier)
    ap.add_argument("--generator", default="extractive", choices=["extractive", "hf", "openrouter"])
    ap.add_argument("--hf-model", default=GuardConfig.hf_model)
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N questions")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def evaluate(args) -> pd.DataFrame:
    corpus, qa = load_dataset(args.data)
    qa = qa[: args.limit] if args.limit else qa
    cfg = GuardConfig(embedder=args.embedder, verifier=args.verifier, generator=args.generator, hf_model=args.hf_model)
    components = Components(cfg, corpus)
    print(f"Embedder: {components.embedder.name} | Generator: {components.generator.name} | "
          f"Verifier: {components.verifier.backend.name if components.verifier else '-'}")

    rows = []
    for n_poison in args.poison_counts:
        poison = [p for item in qa for p in make_poison(item.question, item.target, n_poison, item.id, args.seed)]
        components.reindex(corpus + poison)
        for method, cls in METHODS.items():
            pipe = cls(components)
            for item in qa:
                r = pipe.run(item.question)
                # "targeted" = poison written for THIS question (others are just noise)
                targeted = lambda pid: is_poison(pid) and pid.split("::")[1] == item.id  # noqa: E731
                retrieved_poison = [p["id"] for p in r.passages if targeted(p["id"])]
                flagged = [p["id"] for p in r.passages if p.get("flagged")]
                flagged_poison = [i for i in flagged if is_poison(i)]
                rows.append({
                    "poison_per_question": n_poison,
                    "method": method,
                    "id": item.id,
                    "question": item.question,
                    "answer": r.answer,
                    "mode": r.mode,
                    "quality": r.quality,
                    "correct": contains_answer(r.answer, item.answers),
                    "attacked": contains_answer(r.answer, [item.target]) and not contains_answer(r.answer, item.answers),
                    "abstained": r.answer.strip() == ABSTAIN,
                    "unsupported_rate": r.unsupported_rate,
                    "poison_in_context": sum(targeted(i) for i in r.used_ids),
                    "retrieved_poison": len(retrieved_poison),
                    "flagged": len(flagged),
                    "flagged_poison": len(flagged_poison),
                    "flagged_targeted": sum(targeted(i) for i in flagged),
                    "latency_s": r.latency_s,
                })
            print(f"  poison={n_poison} {method:<12} done")
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["poison_per_question", "method"], sort=False)
    s = g.agg(
        accuracy=("correct", "mean"),
        attack_success_rate=("attacked", "mean"),
        abstain_rate=("abstained", "mean"),
        unsupported_claim_rate=("unsupported_rate", "mean"),
        poison_in_context=("poison_in_context", "mean"),
        avg_latency_ms=("latency_s", lambda x: 1000 * x.mean()),
    ).reset_index()
    guard = df[df["method"] == "ACRC-Guard"].groupby("poison_per_question")
    det = guard.agg(flagged=("flagged", "sum"), flagged_poison=("flagged_poison", "sum"), flagged_targeted=("flagged_targeted", "sum"), retrieved_poison=("retrieved_poison", "sum"))
    det["filter_precision"] = det["flagged_poison"] / det["flagged"].where(det["flagged"] > 0)
    det["filter_recall"] = det["flagged_targeted"] / det["retrieved_poison"].where(det["retrieved_poison"] > 0)
    s = s.merge(det[["filter_precision", "filter_recall"]].reset_index(), on="poison_per_question", how="left")
    s.loc[s["method"] != "ACRC-Guard", ["filter_precision", "filter_recall"]] = None
    return s.round(3)


def plot(summary: pd.DataFrame, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for metric, fname, title in [
        ("attack_success_rate", "attack_success.png", "Attack success rate (lower is better)"),
        ("accuracy", "accuracy.png", "Answer accuracy (higher is better)"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4))
        for method, grp in summary.groupby("method", sort=False):
            ax.plot(grp["poison_per_question"], grp[metric], marker="o", linewidth=2, label=method)
        ax.set_xlabel("Poisoned passages injected per question")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / fname, dpi=150)
        plt.close(fig)


def main():
    logging.basicConfig(level=logging.WARNING)
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = evaluate(args)
    summary = summarize(df)
    df.to_csv(out / "per_question.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "summary.md").write_text(summary.to_markdown(index=False) if _has_tabulate() else summary.to_string(index=False))
    plot(summary, out)
    print("\n" + summary.to_string(index=False))
    print(f"\nSaved results to {out.resolve()}")


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
