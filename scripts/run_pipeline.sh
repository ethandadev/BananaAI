#!/usr/bin/env bash
# The full run, start to finish.
#
#   bash scripts/run_pipeline.sh smoke    # ~10 minutes, tiny model, proves the wiring
#   bash scripts/run_pipeline.sh full     # ~3 days, the real 327M run
#
# Run the smoke pass first. A crash at hour 40 of a run you never rehearsed is
# the most expensive mistake available here.

set -euo pipefail

MODE="${1:-smoke}"
say() { printf "\n\033[1m==> %s\033[0m\n" "$1"; }

if [ ! -d .venv ]; then
  echo "No .venv found. Run: bash scripts/setup_wsl.sh" >&2
  exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

case "$MODE" in
  smoke)
    LIMIT="--limit 20000"
    VOCAB=8192
    VAL_TOKENS=200000
    TRAIN_ARGS="--preset small --steps 200 --warmup 20 --lr 1e-3 --micro-batch 8 \
                --tokens-per-step 16384 --eval-every 50 --ckpt-every 100"
    SFT_ARGS="--epochs 1 --batch-size 4"
    ;;
  full)
    LIMIT=""
    VOCAB=32768
    VAL_TOKENS=10000000
    TRAIN_ARGS=""            # the defaults in core/config.py are the real run
    SFT_ARGS="--epochs 3 --batch-size 8"
    ;;
  *)
    echo "usage: $0 [smoke|full]" >&2
    exit 2
    ;;
esac

say "1/7  Corpus  (streaming, filtering, deduping)"
# shellcheck disable=SC2086
python -m data.prepare --all $LIMIT --out data/processed

say "2/7  Tokenizer"
python -m data.train_tokenizer --shards data/processed --vocab "$VOCAB" --out tokenizer.json

say "3/7  Tokenizing the corpus"
python -m data.tokenize_corpus --shards data/processed --tokenizer tokenizer.json \
    --out data/tokenized --val-tokens "$VAL_TOKENS"

say "4/7  Pretraining"
# shellcheck disable=SC2086
python -m core.train --data data/tokenized --out runs/base --tokenizer tokenizer.json $TRAIN_ARGS

say "5/7  Instruction tuning"
if [ -f data/sft/train.jsonl ]; then
  # shellcheck disable=SC2086
  python -m post.sft --base runs/base/best.pt --data data/sft/train.jsonl \
      --out runs/sft --tokenizer tokenizer.json $SFT_ARGS
else
  echo "    no data/sft/train.jsonl -- skipping SFT and DPO"
  exit 0
fi

say "6/7  Preference tuning"
if [ -f data/dpo/prefs.jsonl ]; then
  python -m post.dpo --sft runs/sft/sft.pt --data data/dpo/prefs.jsonl \
      --out runs/dpo --tokenizer tokenizer.json
  FINAL=runs/dpo/dpo.pt
else
  echo "    no data/dpo/prefs.jsonl -- skipping DPO"
  FINAL=runs/sft/sft.pt
fi

say "7/7  Export"
python -m export.to_safetensors --ckpt "$FINAL" --out export/model --dtype float16
python -m export.to_gguf --ckpt "$FINAL" --tokenizer tokenizer.json \
    --out export/model-f16.gguf

cat <<NOTE

Done. Final checkpoint: $FINAL

Quantise with llama.cpp if you want smaller weights to publish:
    llama-quantize export/model-f16.gguf export/model-Q8_0.gguf   Q8_0
    llama-quantize export/model-f16.gguf export/model-Q4_K_M.gguf Q4_K_M

Talk to it:
    python -m eval.run_eval --ckpt $FINAL --data data/tokenized --chat

NOTE
