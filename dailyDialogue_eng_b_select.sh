export BERT_BASE_DIR=../BERTModels/cased_L-12_H-768_A-12 # The BERT model we are using.

BSIZE=$1

export DDIAL_DIR=../Datasets/ijcnlp_dailydialog # Assumes you have already unzipped the train/test/dev examples. This is to be automated in the future.

export SCRATCH=../../
export OUTDIR=$SCRATCH/ddial_out_eng_b$BSIZE


for i in $(seq 1 5);
do
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
	  --train_batch_size=$BSIZE \
	  --learning_rate=2e-5 \
	  --num_train_epochs=1.0 \
	  --output_dir=$OUTDIR

	python3 run_classifier.py \
	  --task_name=ddial \
	  --twotext=true \
	  --do_train=false\
	  --do_eval=true\
	  --do_predict=false\
	  --do_lower_case=false \
	  --data_dir=$DDIAL_DIR \
	  --vocab_file=$BERT_BASE_DIR/vocab.txt \
	  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
	  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
	  --max_seq_length=60 \
	  --train_batch_size=$BSIZE \
	  --learning_rate=2e-5 \
	  --num_train_epochs=1.0 \
	  --output_dir=$OUTDIR

	python3 run_classifier.py \
	  --task_name=ddial \
	  --twotext=true \
	  --do_train=false\
	  --do_eval=false\
	  --do_predict=true\
	  --do_lower_case=false \
	  --data_dir=$DDIAL_DIR \
	  --vocab_file=$BERT_BASE_DIR/vocab.txt \
	  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
	  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
	  --max_seq_length=60 \
	  --train_batch_size=$BSIZE \
	  --learning_rate=2e-5 \
	  --num_train_epochs=1.0 \
	  --output_dir=$OUTDIR
	sleep 2
	
	mv $OUTDIR/confussion_matrixnorm.png $OUTDIR/confussion_matrixnorm_$i.png
	mv $OUTDIR/confussion_matrixunnorm.png $OUTDIR/confussion_matrixunnorm_$i.png

done
