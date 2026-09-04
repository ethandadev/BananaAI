# BananaAI

Train your own language model, from scratch, on your own computer.

Not a fine-tune of someone else's weights: own tokenizer, own architecture,
own training loop, own sampler. BananaAI looks at your hardware, picks a model
size that will actually finish, and runs the whole pipeline — corpus,
tokenizer, pretraining, instruction tuning, preference tuning, export — from a
desktop app or a single command.

MIT licensed. Nothing leaves your machine.

## What you get

A real base model in the 13M–1.2B range depending on your hardware, exported
as GGUF so llama.cpp, LM Studio, Ollama and friends can load it.

Be clear-eyed about the ceiling: a model of this size is fluent and completes
simple code, but it confabulates facts and fails multi-step reasoning. The
comparison class is GPT-2 and TinyLlama, not a frontier model. What you get
that you cannot buy is a model you built and understand end to end.

## Install

```bash
git clone https://github.com/ethandadev/BananaAI && cd BananaAI
python scripts/setup.py
```

That detects your accelerator, installs the matching PyTorch build, and
verifies it by running real kernels. One setup path for Linux, macOS and
Windows.

Picking the right PyTorch wheel is the most common way this goes wrong: the
wrong one installs cleanly and then fails at the first kernel launch. `setup.py`
reads your driver version and chooses for you.

**Windows:** training is more reliable under WSL2 — `torch.compile` is worth
about 30% throughput and is far steadier on Linux. Install WSL with
`wsl --install -d Ubuntu-24.04`, then run the setup inside it. The desktop app
still builds natively.

## What will my computer train?

```bash
python -m core.hardware
```

```
device     NVIDIA GeForce RTX 5090, 32.6 GB, sm_120, bf16
preset     base  (327M parameters)
context    2048
batch      16 x 16 accum = 524,288 tokens/step
steps      13,400 (7.03B tokens)
memory     25.0 GB of 32.6 GB (weights 4.9 + logits 6.0 + activations 14.1)
estimate   51.1 hours
```

The size is chosen to fit **both** memory and a time budget, because a fully
trained small model beats an undertrained large one. Sizing on memory alone
recommends a 29-day run on a 5090 and a 645M model for a CPU — neither of
which anyone will ever finish.

| Hardware | Picks | Time |
|---|---|---|
| RTX 5090 (32 GB) | base, 327M | ~51 h |
| RTX 4090 (24 GB) | base, 327M | ~65 h |
| RTX 3060 (12 GB) | small, 101M | ~40 h |
| Laptop GPU (8 GB) | small, 101M | ~34 h |
| Apple M-series | mini, 45M | ~29 h |
| CPU only | nano, 13M | ~41 h |

Raise the budget with `--max-hours`, or force a size with `--preset`.

## Run it

```bash
python scripts/run_pipeline.py smoke     # ~15 min, proves the wiring
python scripts/run_pipeline.py full      # the real run
```

Or open the app and press a button:

```bash
cd app && npm install && npm start
```

The **Train** tab shows what your machine will build, which stages to run, a
live loss curve, throughput and time remaining. **Stop & save** finishes the
current step and checkpoints rather than throwing the work away. The **Chat**
tab talks to whatever you have trained.

## Configure

```bash
python -m core.settings --init      # writes bananaai.toml for this machine
```

Everything is in one file: hardware, model size, corpus, batch schedule,
post-training, export. Layered defaults → `bananaai.toml` → `BANANAAI_*`
environment variables → command-line flags.

```toml
[hardware]
device = "auto"        # auto | cuda | mps | cpu
max_hours = 72.0

[model]
preset = "auto"        # auto | nano micro mini small base large xl
vocab_size = 32768

[data]
custom_dir = ""        # a folder of your own documents
```

Unknown sections and misspelled keys are errors, not silent no-ops — a key
that is quietly ignored is a setting you believe is applied.

## Your own data

```bash
python -m data.custom --dir ~/notes --dry-run
```

Reads text, markdown, source code, JSONL and PDF; skips `.git`,
`node_modules`, oversized files and anything whose extension claims text but
whose content is binary. Then it tells you the truth about your folder:

```
  kept  1,284 documents (18.2M characters)
  covers 2.3% of what this model size wants
  too small to pretrain on: pretrain on public data, then fine-tune on this
  folder with post/sft.py
```

Almost every personal folder is orders of magnitude short of a pretraining
corpus. Finding that out beforehand is better than discovering it after two
days of training.

## The pipeline

| Stage | What it does |
|---|---|
| `data` | stream, filter, dedupe and shard a corpus |
| `tokenizer` | train a byte-level BPE on that corpus |
| `tokenize` | encode it to flat `uint16` token bins |
| `pretrain` | the long one — next-token prediction |
| `sft` | instruction tuning, so it answers rather than continues |
| `dpo` | preference tuning; tone and formatting |
| `export` | safetensors and GGUF |

Prose and code get separate quality filters — a rule rejecting documents for a
high symbol-to-word ratio would throw away every source file. Exact duplicates
go via a global hash set, near-duplicates via MinHash + LSH in a sliding
window, because a global LSH index over tens of millions of documents does not
fit in memory.

Training checkpoints every N steps and on every validation improvement.
`--resume` restores the optimizer state, step counter and RNG streams. A loss
watchdog halts on a spike or a non-finite loss: bf16 pretraining diverges from
a single bad batch, and without a halt the run keeps burning hours on NaNs.

## Architecture

| | |
|---|---|
| Layout | RoPE, RMSNorm, SwiGLU, grouped-query attention, tied embeddings |
| Precision | bf16 where supported, fp32 otherwise |
| Vocabulary | byte-level BPE, no `<unk>`, digits split individually |
| Context | 512–2048 depending on size |

The shape matches Llama's deliberately. Every line is written here, but
keeping the layout conventional means the weights export to GGUF without a
custom converter and load with `AutoModelForCausalLM`.

## Tests

```bash
python tests/run_all.py            # 188 tests, ~13s on CPU
python tests/run_all.py --quick    # skip the integration suite
python tests/run_all.py --only export
```

No GPU and no pytest required. The suite targets failures that are otherwise
silent: RoPE translation invariance, causal masking, KV-cache decoding
matching a full forward pass, prompt masking during SFT, the DPO objective's
fixed points, GGUF vocabulary ordering, model sizing on hardware neither of us
owns, and a full corpus-to-generation integration run.

## The target-shift contract

`Transformer.forward` does **not** shift targets. Position `i` predicts
`targets[i]`, so callers pass `x = tokens[:-1]` and `y = tokens[1:]`.

Worth stating loudly because getting it wrong fails quietly: unshifted targets
train the model to copy its own input, and with tied embeddings the residual
stream carries the input embedding straight to the output row, so loss drops
*below* `ln(vocab)` and looks like an unusually good start.

## Layout

```
core/       model, config, tokenizer, dataset, training loop, sampler, hardware, settings
data/       filters, dedupe, corpus pipeline, custom folders, post-training datasets
post/       SFT and DPO
export/     safetensors and GGUF writers
server/     inference sidecar and training sidecar
app/        Electron front end
eval/       perplexity and side-by-side generation
scripts/    setup, GPU verification, the pipeline runner
tests/      188 tests
```

## License

MIT.
