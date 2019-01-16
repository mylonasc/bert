export BERT_BASE_DIR=../BERTModels/multi_cased_L-12_H-768_A-12 # The BERT model we are using.

#export BERT_BASE_DIR=../BERTModels/cased_L-12_H-768_A-12 # The BERT model we are using.

#export GLUE_DIR=../glue_data
export DATASET_PATH=../Datasets/ijcnlp_dailydialog  # Assumes you have already unzipped the train/test/dev examples. This is to be automated in the future.
export OUTDIR=../../ddial_out
BSIZE=32


python3 show_attention.py \
  --task_name=ddial\
  --twotext=true \
  --do_train=false\
  --do_eval=false\
  --do_predict=true\
  --do_lower_case=false \
  --data_dir=$DATASET_PATH\
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$OUTDIR/model.ckpt-1000 \
  --diff_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=60 \
  --train_batch_size=$BSIZE \
  --learning_rate=2e-5 \
  --num_train_epochs=1.0 \
  --output_dir=$OUTDIR
