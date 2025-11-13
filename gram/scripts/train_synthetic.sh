#!/bin/bash
# train_pretrain.sh
# ------------------------------------------------------------
# Train GRAM on synthetic data (pretraining step).
#
# Default:
#   bash train_pretrain.sh
#
# Custom:
#   bash train_pretrain.sh synthetic_tree.seqs synthetic_tree.seqs synthetic_tree synthetic_pretrain
# ------------------------------------------------------------

set -e

SEQ_FILE=${1:-synthetic_tree.seqs}
LABEL_FILE=${2:-synthetic_tree.seqs}   # GRAM auto shifts next-visit internally
TREE_PREFIX=${3:-synthetic_tree}
OUT_PREFIX=${4:-synthetic_pretrain}

echo "=== PRETRAIN GRAM ON SYNTHETIC ==="
echo "Seq file:   $SEQ_FILE"
echo "Label file: $LABEL_FILE"
echo "Tree:       $TREE_PREFIX"
echo "Output:     $OUT_PREFIX"
echo

python3 gram.py \
    "$SEQ_FILE" \
    "$LABEL_FILE" \
    "$TREE_PREFIX" \
    "$OUT_PREFIX" \
    --embed_size 128 \
    --rnn_size 128 \
    --attention_size 128 \
    --batch_size 64 \
    --n_epochs 20 \
    --L2 0.001 \
    --dropout_rate 0.5 \
    --log_eps 1e-8

echo
echo "=== DONE: PRETRAIN FINISHED ==="
echo "Checkpoints saved as: ${OUT_PREFIX}.*.npz"
echo "Log file: ${OUT_PREFIX}.log"
