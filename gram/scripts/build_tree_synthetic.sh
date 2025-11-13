#!/bin/bash
# build_tree_synthetic.sh
# ------------------------------------------------------------
# Build CCS ancestor tree for synthetic GRAM data.
#
# Usage:
#   bash build_tree_synthetic.sh \
#       ccs_multi_dx_tool_2015.csv \
#       synthetic.seqs \
#       synthetic.types \
#       synthetic_tree
# ------------------------------------------------------------

set -e

CCS_FILE=${1:-ccs_multi_dx_tool_2015.csv}
SEQ_FILE=${2:-synthetic.seqs}
TYPE_FILE=${3:-synthetic.types}
OUT_PREFIX=${4:-synthetic_tree}

echo "=== BUILD TREE FOR SYNTHETIC DATA ==="
echo "CCS file:   $CCS_FILE"
echo "Seq file:   $SEQ_FILE"
echo "Types file: $TYPE_FILE"
echo "Out prefix: $OUT_PREFIX"
echo

# Kaggle uses Python 3 by default
python3 ../model/build_trees.py ... "$CCS_FILE" "$SEQ_FILE" "$TYPE_FILE" "$OUT_PREFIX"

echo
echo "=== DONE ==="
echo "Created:"
echo "  $OUT_PREFIX.level1.pk"
echo "  $OUT_PREFIX.level2.pk"
echo "  $OUT_PREFIX.level3.pk"
echo "  $OUT_PREFIX.level4.pk"
echo "  $OUT_PREFIX.level5.pk"
echo "  $OUT_PREFIX.types"
echo "  $OUT_PREFIX.seqs"
