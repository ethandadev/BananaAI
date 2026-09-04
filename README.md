# A 327M foundation model, trained from scratch on one GPU

A decoder-only language model built end to end: own BPE tokenizer, own
architecture, own training loop, own sampler. Pretrained on 7B tokens on a
single RTX 5090, then instruction-tuned and preference-tuned, and shipped as a
desktop app with GGUF weights.

Nothing here is fine-tuned from someone else's base model.

## Specification

| | |
|---|---|
| Parameters | 326,685,696 (26 layers, d_model 1024, 16 heads / 4 KV heads) |
| Training tokens | 7.0B (~21 per parameter) |
| Context | 2048 |
| Vocabulary | 32,768 byte-level BPE, trained on this corpus |
| Architecture | RoPE, RMSNorm, SwiGLU, grouped-query attention, tied embeddings |
| Precision | bf16 with fp32 master weights |
| Batch | 524,288 tokens/step (micro 16 × 16 grad-accum × 2048) |
| Hardware | 1 × RTX 5090 (32 GB) |
| Pretrain wall time | ~52 hours |

The architecture matches Llama's layout on purpose. Every line is written
here, but keeping the shape conventional means the weights export to GGUF
without a custom converter, and load with `AutoModelForCausalLM`.

## Layout

```
core/       model, config, dataset, tokenizer, training loop, sampler
data/       filters, dedupe, corpus pipeline, tokenizer training
post/       SFT and DPO
export/     safetensors and GGUF writers
server/     stdio JSON sidecar for the app
app/        Electron front end
eval/       perplexity and side-by-side generation
scripts/    environment setup, GPU verification, the full runbook
tests/      96 tests, no GPU and no pytest required
```

## Getting started

Training runs under WSL2 Ubuntu; the desktop app builds natively on Windows.
`torch.compile` is meaningfully more reliable on Linux, and losing it costs
roughly 30% throughput — about 16 hours on a 52-hour run.

```bash
git clone <this repo> ~/llm && cd ~/llm
bash scripts/setup_wsl.sh
```

That installs the toolchain, creates a venv, pulls the CUDA 12.8 PyTorch build
(Blackwell is `sm_120`; older wheels install cleanly and then fail at the first
kernel launch), and runs the verification suite:

```bash
python scripts/verify_env.py
```

All eight checks must pass before starting a training run. They run real
kernels rather than trusting `torch.cuda.is_available()`, and the last one
builds the actual model and checks that initial loss lands near
`ln(32768) = 10.4` — which catches a bad initialisation in 30 seconds instead
of 40 hours in.

Keep the repo and the data on the WSL filesystem, not `/mnt/c`. Small random
reads across the 9p boundary are about 10× slower, and a dataloader does
nothing but small random reads.

## Running the pipeline

```bash
bash scripts/run_pipeline.sh smoke    # ~10 min, small model, proves the wiring
bash scripts/run_pipeline.sh full     # ~3 days, the real run
```

Run the smoke pass first. Or drive the stages individually:

```bash
python -m data.prepare --all --out data/processed
python -m data.train_tokenizer --shards data/processed --vocab 32768
python -m data.tokenize_corpus --shards data/processed --out data/tokenized
python -m core.train --data data/tokenized --out runs/base
python -m post.sft --base runs/base/best.pt --data data/sft/train.jsonl --out runs/sft
python -m post.dpo --sft runs/sft/sft.pt --data data/dpo/prefs.jsonl --out runs/dpo
python -m export.to_gguf --ckpt runs/dpo/dpo.pt --out export/model-f16.gguf
```

Training checkpoints every 1000 steps and on every validation improvement.
`--resume` picks up from the latest checkpoint with the optimizer state, step
counter, and RNG streams intact. Ctrl-C finishes the current step and
checkpoints before exiting rather than losing it.

The loss watchdog halts on a spike or a non-finite loss. bf16 pretraining
diverges from a single bad batch, and without a halt the run keeps burning GPU
hours producing NaNs.

## Corpus

| Source | Share | Kind |
|---|---|---|
| FineWeb-Edu | 55% | prose |
| The Stack v2 | 25% | code |
| Wikipedia (en) | 8% | prose |
| StackExchange | 7% | prose |
| OpenWebMath | 5% | prose |

Prose and code get different quality filters — a rule that rejects documents
for a high symbol-to-word ratio would throw away every source file. Exact
duplicates are removed globally with a hash set; near-duplicates with MinHash
+ LSH inside a sliding window, because a global LSH index over tens of
millions of documents does not fit in 32 GB. The manifest records what each
stage dropped and why.

## Tests

```bash
python tests/run_all.py            # 96 tests, ~12s on CPU
python tests/run_all.py --quick    # skip the integration suite
python tests/run_all.py --only export
```

No GPU and no pytest needed. Coverage includes the failures that stay silent:
RoPE translation invariance, causal masking, incremental KV-cache decoding
matching a full forward pass, prompt masking during SFT, the DPO objective's
fixed points, GGUF vocabulary ordering, and a full corpus-to-generation
integration run.

## The target-shift contract

`Transformer.forward` does **not** shift targets. Position `i` predicts
`targets[i]`, so callers pass `x = tokens[:-1]` and `y = tokens[1:]`.

This is worth stating loudly because getting it wrong fails quietly. Unshifted
targets train the model to copy its own input — and with tied embeddings the
residual stream carries the input embedding straight to the output row, so the
loss drops *below* `ln(vocab)` and looks like an unusually good start rather
than a bug.

## Desktop app

Electron shell, Python sidecar, newline-delimited JSON over stdio. The
renderer reaches the model only through a preload bridge with
`contextIsolation` on, and model output is inserted with `textContent` — a
generated `<img onerror>` must never execute inside the app.

```bash
cd app && npm install && npm start
```

The sidecar is usable on its own, which is what makes the UI replaceable:

```bash
echo '{"id":"1","type":"generate","messages":[{"role":"user","content":"hi"}]}' \
  | python -m server.sidecar --ckpt runs/dpo/dpo.pt
```

## Status

- [x] Model, config, environment verification
- [x] Data pipeline — filters, dedupe, sharding
- [x] Tokenizer — byte-level BPE with reserved chat tokens
- [x] Pretraining — checkpointing, resume, watchdog, MFU
- [x] SFT and DPO
- [x] Sampler with KV cache; safetensors and GGUF export
- [x] Sidecar and Electron front end
- [ ] A real training run

## Expectations

At 327M parameters the model will be fluent and will complete simple code. It
will confabulate facts and fail multi-step reasoning. That is the ceiling at
this scale, not a defect. The comparison class is GPT-2 and TinyLlama.

## License

MIT.
