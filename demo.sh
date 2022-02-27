#!/bin/bash
set -e
make

USE_DEF_LOSS=$1
LABMDA=$2
USE_ORTHO_LOSS=$3
BETA=$4
USE_PROJ_LOSS=$5
GAMMA=$6
MAX_ITER=$7

VERBOSE=2
MEMORY=4.0
VOCAB_MIN_COUNT=5
MAX_VOCAB=400000
VECTOR_SIZE=300
WINDOW_SIZE=10
BINARY=2
NUM_THREADS=32
X_MAX=100

CORPUS=text8
VOCAB_FILE=vocab.txt
COOCCURRENCE_FILE=cooccurrence.bin
COOCCURRENCE_SHUF_FILE=cooccurrence.shuf.bin
BUILDDIR=build
SAVE_FILE=vectors_l1def$1_labmda$2_ortho$3_beta$4_proj$5_gamma$6_itr$MAX_ITER
VECTOR_TEXT_FILE=$SAVE_FILE.txt


if hash python 2>/dev/null; then
    PYTHON=python
else
    PYTHON=python3
fi

echo
echo "$ $BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -max-vocab $MAX_VOCAB -verbose $VERBOSE < $CORPUS > $VOCAB_FILE"
$BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -max-vocab $MAX_VOCAB -verbose $VERBOSE < $CORPUS > $VOCAB_FILE
echo "$ $BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE"
$BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE
echo "$ $BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE"
$BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE
echo "$ $PYTHON get_definition.py --vocab_file $VOCAB_FILE --max_vocab_size $MAX_VOCAB"
$PYTHON get_definition.py --vocab_file $VOCAB_FILE --max_vocab_size $MAX_VOCAB

echo "$ $BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX \
-iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE \
-use-def-loss $USE_DEF_LOSS -lambda $LABMDA -use-ortho-loss $USE_ORTHO_LOSS -beta $BETA -use-proj-loss $USE_PROJ_LOSS -gamma $GAMMA -seed 10"
$BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX \
-iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE -write-header 1 \
-use-def-loss $USE_DEF_LOSS -lambda $LABMDA -use-ortho-loss $USE_ORTHO_LOSS -beta $BETA -use-proj-loss $USE_PROJ_LOSS -gamma $GAMMA -seed 10 #-cechkpoint-every 1

# quick evaluation of word embedding, including Google analogy test and WEAT
echo
echo "$ $PYTHON eval/python/evaluate.py --vocab_file $VOCAB_FILE --vectors_file $VECTOR_TEXT_FILE"
$PYTHON eval/python/evaluate.py --vocab_file $VOCAB_FILE --vectors_file $VECTOR_TEXT_FILE