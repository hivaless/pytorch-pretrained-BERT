export PYTHONPATH=`pwd`

python examples/lm_finetuning/finetune_on_pregenerated.py \
    --pregenerated_data=training/ \
    --bert_config=samples/bert_config.json \
    --output_dir=finetuned_lm/ \
    --epochs=4 \
    --train_batch_size=32 \
    --vocab_file=samples/hangul_vocab.txt
