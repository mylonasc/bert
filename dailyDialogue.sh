export BERT_BASE_DIR=../BERTModels/multi_cased_L-12_H-768_A-12 # The BERT model we are using.
#export GLUE_DIR=../glue_data
export DDIAL_DIR=../Datasets/ijcnlp_dailydialog # Assumes you have already unzipped the train/test/dev examples. This is to be automated in the future.
export OUTDIR=ddial_out_$1


python3 run_classifier.py \
  --task_name=ddial \
  --twotext=true \
  --do_train=true\
  --do_eval=false\
  --do_predict=false\
  --do_lower_case=false \
  --data_dir=$DDIAL_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=60 \
  --train_batch_size=32 \
  --learning_rate=1e-5 \
  --num_train_epochs=1.0 \
  --output_dir=/tmp/$OUTDIR
