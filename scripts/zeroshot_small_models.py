"""Zero-shot Telugu->Telugu summarization with small (<1B) Indian models.

generate: run IndicBART-family seq2seq checkpoints (or an extractive lead baseline)
          on the 100 synthetic notes and write results.jsonl.
score:    score any results.jsonl (including vemana_study.cli runs) with the same
          mechanical diagnostics. None of these establish factual accuracy.
"""

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # vemana_study and engine live at the repo root
DATASET = ROOT / "benchmark_100" / "dataset.json"
NOTES = ROOT / "benchmark_100" / "input_notes"

MODELS = {
    "indicbart-xlsum": "ai4bharat/IndicBART-XLSum",
    "sentsum-ss": "ai4bharat/MultiIndicSentenceSummarizationSS",
}


def load_notes():
    meta = json.loads(DATASET.read_text(encoding="utf-8"))["notes"]
    return [(m, (NOTES / m["file"]).read_text(encoding="utf-8").strip()) for m in meta]


def fresh(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise SystemExit(f"{path} exists and is not empty; refusing to overwrite")
    path.mkdir(parents=True, exist_ok=True)
    return path


def body_paragraphs(source):
    return [p.strip() for p in source.split("\n\n") if p.strip()]


def lead_summary(source):
    """Extractive baseline: the first paragraph after the administrative header."""
    paras = body_paragraphs(source)
    return paras[1] if len(paras) > 1 else paras[0]


def generate(args):
    import torch
    from transformers import AlbertTokenizer, AutoModelForSeq2SeqLM
    import transformers

    out = fresh(args.output_dir)
    notes = load_notes()
    run = {"model": args.model, "device": args.device, "num_beams": args.num_beams,
           "max_length": args.max_length, "no_repeat_ngram_size": args.no_repeat_ngram_size,
           "batch_size": args.batch_size, "dtype": "float32",
           "torch": torch.__version__, "transformers": transformers.__version__,
           "factual_accuracy": None, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    records = []
    t_load = time.time()
    if args.model == "lead":
        for meta, source in notes:
            records.append({"id": meta["file"], "output": lead_summary(source), "ended": True,
                            "hit_limit": False, "input_tokens": None, "truncated": False,
                            "generated_tokens": None, "seconds": 0.0})
        run["load_seconds"] = 0.0
    else:
        repo = MODELS[args.model]
        tok = AlbertTokenizer.from_pretrained(repo, do_lower_case=False, use_fast=False, keep_accents=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(repo).eval().to(args.device)
        ids = {s: tok.convert_tokens_to_ids(s) for s in ["<s>", "</s>", "<pad>", "<2te>"]}
        run.update(repo=repo, parameters=sum(p.numel() for p in model.parameters()), special_ids=ids)
        run["load_seconds"] = round(time.time() - t_load, 2)
        for start in range(0, len(notes), args.batch_size):
            batch = notes[start:start + args.batch_size]
            texts = [src + " </s> <2te>" for _, src in batch]
            enc = [tok(t, add_special_tokens=False).input_ids for t in texts]
            lengths = [len(e) for e in enc]
            # Keep the language tag at the end if truncation is ever needed.
            enc = [e if len(e) <= 1024 else e[:1022] + e[-2:] for e in enc]
            width = max(len(e) for e in enc)
            input_ids = torch.full((len(enc), width), ids["<pad>"], dtype=torch.long)
            mask = torch.zeros_like(input_ids)
            for i, e in enumerate(enc):
                input_ids[i, :len(e)] = torch.tensor(e)
                mask[i, :len(e)] = 1
            if args.device == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                gen = model.generate(input_ids=input_ids.to(args.device), attention_mask=mask.to(args.device),
                                     num_beams=args.num_beams, max_length=args.max_length,
                                     early_stopping=True, no_repeat_ngram_size=args.no_repeat_ngram_size,
                                     pad_token_id=ids["<pad>"], bos_token_id=ids["<s>"],
                                     eos_token_id=ids["</s>"], decoder_start_token_id=ids["<2te>"])
            if args.device == "cuda":
                torch.cuda.synchronize()
            per_note = (time.time() - t0) / len(batch)
            for (meta, _), row, n in zip(batch, gen.tolist(), lengths):
                row = row[1:]  # drop decoder start (<2te>)
                ended = ids["</s>"] in row
                if ended:
                    row = row[:row.index(ids["</s>"])]
                row = [t for t in row if t != ids["<pad>"]]
                text = tok.decode(row, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
                text = re.sub(r"<2[a-z]+>", "", text).strip()
                records.append({"id": meta["file"], "output": text, "ended": ended,
                                "hit_limit": (len(row) + 2) >= args.max_length,
                                "input_tokens": n, "truncated": n > 1024,
                                "generated_tokens": len(row), "seconds": round(per_note, 4)})
            print(f"{start + len(batch)}/{len(notes)}", flush=True)
        if args.device == "cuda":
            run["peak_cuda_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    with (out / "results.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    run.update(status="completed", finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               generation_seconds_total=round(sum(r["seconds"] for r in records), 2))
    (out / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out}")


def words(text):
    return re.findall(r"[ఀ-౿]+|[A-Za-z0-9-]+", text)


def score(args):
    from sacrebleu.metrics import CHRF
    from vemana_study.quality import check_output

    chrf = CHRF(word_order=2)  # chrF++
    notes = {m["file"]: (m, s) for m, s in load_notes()}
    rows = [json.loads(l) for l in Path(args.results).read_text(encoding="utf-8").splitlines() if l.strip()]
    scored = []
    for r in rows:
        meta, source = notes[r["id"]]
        output = r.get("output", "")
        if "stopped_at_eos" in r:  # vemana_study.cli record format
            ended = r["stopped_at_eos"]
            hit_limit = r["generated_tokens"] >= r["max_new_tokens"]
        else:
            ended, hit_limit = r["ended"], r["hit_limit"]
        checks = check_output(source, output, "te", ended=bool(ended), hit_limit=bool(hit_limit),
                              examples=args.examples)
        subject = re.search(r"విషయం (.+?)\.", source)
        src_words = set(words(source))
        out_words = words(output)
        novel = [w for w in out_words if w not in src_words]
        # Prompt scaffolding from vemana_study/prompts.py ("complaint:", "summary:", "write concisely").
        prompt_leak = any(k in output for k in ["సారాంశం", "ఫిర్యాదు:", "సంక్షిప్తంగా", "Summary", "ఉదాహరణ"])
        windows = [tuple(out_words[i:i + 5]) for i in range(len(out_words) - 4)]
        src_list = words(source)
        src_windows = {tuple(src_list[i:i + 5]) for i in range(len(src_list) - 4)}
        copy_rate = sum(w in src_windows for w in windows) / len(windows) if windows else 0.0
        scored.append({
            "id": meta["id"], "category": meta["category"], "urgent": meta["urgent"],
            "basic_checks_passed": not checks["errors"], "errors": ";".join(checks["errors"]),
            "chrf_main_fact": round(chrf.sentence_score(output, [meta["main_fact"]]).score, 2),
            "chrf_subject": round(chrf.sentence_score(output, [subject.group(1)]).score, 2) if subject else None,
            "novel_word_rate": round(len(novel) / len(out_words), 3) if out_words else None,
            "output_words": len(out_words), "prompt_leak": prompt_leak,
            "source_copy_rate": round(copy_rate, 3),
            "urgent_hazard_mentioned": ("విద్యుత్" in output and ("తీగ" in output or "ప్రమాద" in output))
            if meta["urgent"] else None,
            "seconds": r.get("seconds"), "output": output,
        })
    out = Path(args.out_dir or Path(args.results).parent)
    with (out / "scored.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(scored[0]))
        w.writeheader()
        w.writerows(scored)

    def mean(key):
        vals = [s[key] for s in scored if s[key] is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    urgent = [s for s in scored if s["urgent"]]
    summary = {
        "results": str(args.results), "notes": len(scored),
        "basic_checks_passed": sum(s["basic_checks_passed"] for s in scored),
        "mean_chrf_main_fact": mean("chrf_main_fact"), "mean_chrf_subject": mean("chrf_subject"),
        "mean_novel_word_rate": mean("novel_word_rate"), "mean_output_words": mean("output_words"),
        "mean_source_copy_rate": mean("source_copy_rate"),
        "prompt_leaks": sum(s["prompt_leak"] for s in scored),
        "distinct_outputs": len({s["output"] for s in scored}),
        "urgent_hazard_mentioned": [s["urgent_hazard_mentioned"] for s in urgent],
        "urgent_output": [s["output"] for s in urgent],
        "error_counts": {e: sum(e in s["errors"].split(";") for s in scored)
                         for e in sorted({e for s in scored for e in s["errors"].split(";") if e})},
        "factual_accuracy": None,
        "caveat": "Mechanical diagnostics on synthetic templated notes; not factual accuracy.",
    }
    (out / "score_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "urgent_output"}, ensure_ascii=False))
    print("URGENT:", summary["urgent_output"])


REVIEW_FIELDS = ["faithful_yes_no", "unsupported_claims", "critical_omissions",
                 "urgency_preserved_yes_no", "fluency_1_to_5", "reviewer", "reviewer_notes"]


def review_sheet(args):
    """Long-format worksheet: one row per (note, model). Priority file = first note per category."""
    notes = {m["file"]: (m, s) for m, s in load_notes()}
    runs = [tuple(spec.split("=", 1)) for spec in args.run]  # label=path/to/results.jsonl
    outputs = {label: {json.loads(l)["id"]: json.loads(l)["output"]
                       for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()}
               for label, path in runs}
    first_per_category, seen = set(), set()
    for name, (meta, _) in notes.items():
        if meta["category"] not in seen:
            seen.add(meta["category"])
            first_per_category.add(name)
    fields = ["note_id", "category", "urgent", "model", "source_te", "key_fact_te", "summary_te", *REVIEW_FIELDS]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for filename, keep in [("review_priority.csv", first_per_category), ("review_all.csv", set(notes))]:
        with (out / filename).open("w", encoding="utf-8-sig", newline="") as f:  # BOM: Telugu opens in Excel
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for name, (meta, source) in notes.items():
                if name not in keep:
                    continue
                for label, _ in runs:
                    w.writerow({"note_id": meta["id"], "category": meta["category"], "urgent": meta["urgent"],
                                "model": label, "source_te": source, "key_fact_te": meta["main_fact"],
                                "summary_te": outputs[label].get(name, ""), **{k: "" for k in REVIEW_FIELDS}})
    print(f"wrote {out / 'review_priority.csv'} and {out / 'review_all.csv'}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--model", choices=[*MODELS, "lead"], required=True)
    g.add_argument("--output-dir", required=True)
    g.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    g.add_argument("--batch-size", type=int, default=16)
    g.add_argument("--num-beams", type=int, default=4)
    g.add_argument("--max-length", type=int, default=128)
    g.add_argument("--no-repeat-ngram-size", type=int, default=3)
    s = sub.add_parser("score")
    s.add_argument("--results", required=True)
    s.add_argument("--out-dir")
    s.add_argument("--examples", type=int, default=0, help="Prompt examples used (for copy checks)")
    r = sub.add_parser("review-sheet")
    r.add_argument("--run", action="append", required=True, help="label=path/to/results.jsonl (repeatable)")
    r.add_argument("--out-dir", required=True)
    a = p.parse_args()
    {"generate": generate, "score": score, "review-sheet": review_sheet}[a.cmd](a)


if __name__ == "__main__":
    main()
