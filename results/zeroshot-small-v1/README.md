# Zero-shot Telugu summaries with small Indian models (< 1B parameters)

**Run date:** 10 November 2026 · **Hardware:** one NVIDIA H100 80 GB · **Inputs:** the 100 synthetic notes in
[`benchmark_100/`](../../benchmark_100) · **Task:** Telugu note → short Telugu summary · **Code:**
[`scripts/zeroshot_small_models.py`](../../scripts/zeroshot_small_models.py) on base commit `82bedc1`.

**Bottom line:** none of the tested models is usable as-is. AI4Bharat's **IndicBART-XLSum** (244M) is the best starting
point: it always writes fluent Telugu on the right topic and never copies the prompt, but it writes news-headline-style
one-liners and invents or changes details in several categories. It needs fine-tuning on complaint→summary pairs.
**These are mechanical diagnostics, not factual-accuracy scores.** A native Telugu reviewer must confirm the findings.

## Models tested

| Label | Model | Provider | Params | Type | Licence |
|---|---|---|---|---|---|
| `indicbart-xlsum` | [ai4bharat/IndicBART-XLSum](https://huggingface.co/ai4bharat/IndicBART-XLSum) | AI4Bharat (IIT Madras) | 244M | encoder-decoder, fine-tuned on XL-Sum news summaries | base IndicBART is MIT; **XL-Sum data is CC BY-NC-SA 4.0** |
| `sentsum-ss` | [ai4bharat/MultiIndicSentenceSummarizationSS](https://huggingface.co/ai4bharat/MultiIndicSentenceSummarizationSS) | AI4Bharat | 244M | encoder-decoder, sentence-level summaries | MIT |
| `lead` | no model: copies the first paragraph after the header | – | – | extractive sanity baseline | – |

No Sarvam AI model is under 1B parameters (smallest: Sarvam-1, 2B), so none was tested.

## Results

| Model | Basic checks passed | chrF++ vs key fact | chrF++ vs subject line | Avg. words | Novel-word rate | Source-copy rate | Prompt leaked | Urgent wire note | GPU time / note |
|---|---|---|---|---|---|---|---|---|---|
| `indicbart-xlsum` | 100/100 | 22.2 | 46.4 | 7.6 | 0.31 | 0.08 | 0/100 | mentions wires, but **rewritten as a question**, danger to children dropped | ≈0.07 s (batch 16) |
| `sentsum-ss` | 100/100 | 16.8 | 43.4 | 4.5 | 0.28 | 0.04 | 0/100 | "wires fell at the school gate"; danger and request dropped | ≈0.01 s (batch 16) |
| `lead` (copy baseline) | 100/100 | 100.0 | 39.2 | 10.4 | 0.00 | 1.00 | 0/100 | copied verbatim | – |

Column meanings:

- **Basic checks:** the repo's mechanical checks (`vemana_study/quality.py`): non-empty, ≥80% Telugu script, ≤120 words,
  no heavy repetition, no copying of prompt examples. Passing them says nothing about correctness.
- **chrF++ vs key fact:** character n-gram overlap (0–100) with the note's `main_fact` sentence from `dataset.json`.
- **chrF++ vs subject line:** overlap with the `విషయం …` (subject) line of the note header; a rough topic signal.
- **Novel-word rate:** share of summary words that never appear in the source. A rough hallucination signal; Telugu
  inflection inflates it, so read it together with the outputs.
- **Source-copy rate:** share of the summary's 5-word windows that appear verbatim in the source (1.0 = pure copy).
- **Prompt leaked:** the output contains prompt scaffolding such as `ఫిర్యాదు:` (complaint:) or `సారాంశం` (summary).

### What the IndicBART-XLSum outputs look like (first note of each category)

Translations are ours and need native confirmation.

| Category | Output (gloss) | Assessment (to be confirmed) |
|---|---|---|
| water | "Drinking-water problem: village-001, elderly struggle to walk far" | on topic, partial |
| pension | "Why the delay in old-age pension payment?" | on topic, headline/question form |
| roads | "Road repairs: no doubt in our application that someone was injured in road accidents" | **invented injuries** |
| electricity | "Electricity problem: apply online" | **invented / wrong** |
| sanitation | copies the header subject line | on topic, no detail |
| healthcare | "Improve medicine supply and medical services in **old-age centres**" | **wrong place** (health centre) |
| ration | "Ration distribution: delay in **ration card** distribution" | **wrong item** (goods, not cards) |
| land | "Land-record correction: records should be examined and recorded" | roughly right |
| education | copies "subject: school facilities" | loses the broken-toilets detail |
| urgent wires | "Have the electric wires fallen near the school gate?" | **statement turned into a question; urgency lost** |

## Important limits of this benchmark

1. **Only 10 distinct key facts.** The 100 notes are templated: every note in a category shares the same `main_fact`
   (e.g. all 20 water notes). This is effectively a 10-case test, not 100.
2. **The key fact is pasted verbatim as paragraph 2**, so the copy baseline scores a perfect 100. Overlap metrics on
   these notes cannot separate a good summarizer from a copier. Real or deliberately varied complaints are needed.
3. Only one urgent note (`N100`). Urgency handling is not measured statistically.
4. No human has yet judged faithfulness. All factual statements above are provisional.

## What needs review

Please review in this order. Files are UTF-8 with BOM, so Telugu opens correctly in Excel and Google Sheets.

1. **Telugu output quality (native speaker, about 30–45 minutes).** Open [`review/review_priority.csv`](review/review_priority.csv):
   the first note of every category × every model (10 notes × 3 runs = 30 rows). For each row, fill in
   - `faithful_yes_no`: does the summary only state things that are in the source?
   - `unsupported_claims`: anything added or changed (copy the Telugu words).
   - `critical_omissions`: the main problem or requested action missing?
   - `urgency_preserved_yes_no`: for `N100`, is the danger and the need for immediate action kept?
   - `fluency_1_to_5`: natural, grammatical Telugu?
   - `reviewer`, `reviewer_notes`.

   Also confirm or correct the glosses and assessments in the table above.
   [`review/review_all.csv`](review/review_all.csv) has all 300 rows if time allows.
2. **Model choice (supervisor).** Do you agree that IndicBART is the starting point for fine-tuning? Is the < 1B limit
   firm, or is a ~2B fallback (Sarvam-1) acceptable?
3. **Licensing (supervisor / client).** IndicBART-XLSum was trained on CC BY-NC-SA data. The plan is to fine-tune from
   MIT-licensed IndicBARTSS for anything deployed, and use the XL-Sum checkpoint only as a baseline. Is that acceptable?
4. **Benchmark (supervisor).** Agree that the next evaluation set must use varied complaints (ideally 50–100 anonymized
   real ones from the client) rather than the templated notes?
5. **Code (engineer, optional).** [`scripts/zeroshot_small_models.py`](../../scripts/zeroshot_small_models.py): tokenisation
   was checked to match raw SentencePiece exactly (input tokens 537 + 2 tags, no `<unk>`), and no input was truncated
   (all ≤ 1,024 tokens). Decoding settings are in each `run.json`.

## Reproduce

```bash
# from the repo root, with transformers 5.22, torch 2.10, sentencepiece, protobuf, sacrebleu
export HF_HUB_OFFLINE=1   # after downloading the models once with `hf download <repo>`
python scripts/zeroshot_small_models.py generate --model indicbart-xlsum --device cuda --output-dir results/zeroshot-small-v1/indicbart-xlsum
python scripts/zeroshot_small_models.py generate --model sentsum-ss      --device cuda --output-dir results/zeroshot-small-v1/sentsum-ss
python scripts/zeroshot_small_models.py generate --model lead                          --output-dir results/zeroshot-small-v1/lead
python scripts/zeroshot_small_models.py score --results results/zeroshot-small-v1/indicbart-xlsum/results.jsonl
python scripts/zeroshot_small_models.py review-sheet --out-dir results/zeroshot-small-v1/review \
       --run indicbart-xlsum=results/zeroshot-small-v1/indicbart-xlsum/results.jsonl \
       --run sentsum-ss=results/zeroshot-small-v1/sentsum-ss/results.jsonl \
       --run lead=results/zeroshot-small-v1/lead/results.jsonl
```

Decoding for IndicBART models: beam search (4 beams), max 128 tokens, `no_repeat_ngram_size=3`, FP32, batch 16.
The exact commands used are in [`operator/run_zeroshot.sh`](operator/run_zeroshot.sh); `operator/gpu.txt` lists the
node's GPUs (the run was pinned to UUID `GPU-2544ad3e…`).

## Files

| Path | Contents |
|---|---|
| `<model>/results.jsonl` | raw outputs with token counts and timing |
| `<model>/run.json` | settings, versions, parameter count, peak GPU memory |
| `<model>/scored.csv`, `<model>/score_summary.json` | per-note and aggregate diagnostics |
| `review/review_priority.csv`, `review/review_all.csv` | human-review worksheets |

## Next step

Phase 2: build varied complaint→summary training pairs (large model drafts, native speakers correct), then fine-tune
IndicBARTSS on them and evaluate on a held-out, human-reviewed set.
