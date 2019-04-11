export DHA_DIR=/data/project/data/dha/dha_src-2.8.18-x86_64-redhat-linux-gnu
export DHA_RES_DIR=/data/project/data/dha/dha_resources-2.8.106-default
export PYTHONPATH=`pwd`:$DHA_DIR/lib

python3 examples/lm_finetuning/pregenerate_training_data.py \
    --train_corpus=/data/project/data/hangul/brunch.txt \
    --output_dir=training/ \
    --epochs_to_generate=4 \
    --max_seq_len=128 \
    --vocab_file=samples/hangul_vocab.txt \
    --dha_max_workers=16
