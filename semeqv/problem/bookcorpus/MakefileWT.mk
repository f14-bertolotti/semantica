NUM_WORKERS=3
RESTOREPATH=""
RESTOREEPOCH=-1
SEED=42
EPOCHS=30
DEVICE="cuda:0"
COMPILE=False
ETC=1
LR=0.0005
SIZE=128
TRAIN_BATCH_SIZE=200

EMBEDDING_SIZE=128
FEEDFORWARD_SIZE=512
LAYERS=2
HEADS=2
VOCABSIZE=32768
ACTIVATION="gelu"
DROPOUT=0.1

data/bookcorpus/tokenizer.json:
	mkdir -p $(dir $@)
	PYTHONPATH=:.:semeqv \
	python3 semeqv/cli.py \
	cli bookcorpus base-dataset \
	cli train-tokenizer \
		--path $@ \
		--vocab_size ${VOCABSIZE}

data/bookcorpus/seed%/modellast.pt: data/bookcorpus/tokenizer.json
	mkdir -p $(dir $@)
	PYTHONPATH=:.:semeqv \
	python3 semeqv/cli.py \
		--seed $* \
		--compile ${COMPILE} \
		--epochs ${EPOCHS} \
		--trainbar True \
		--validbar True \
		--epochbar False \
	cli callbacks embedding traincallback \
		--path $(dir $@)embeddings.pkl \
		--stc 1000 \
		--step_log_path  $(dir $@)train_step.log \
		--epoch_log_path $(dir $@)train_epoch.log \
	cli callbacks log validcallback \
		--step_log_path  $(dir $@)valid_step.log \
		--epoch_log_path $(dir $@)valid_epoch.log \
	cli loss cross-entropy \
	cli bookcorpus default-dataset \
		--path data/bookcorpus/tokenizer.json \
		--batch_size ${TRAIN_BATCH_SIZE} \
		--drop_last True \
		--size ${SIZE} \
		--device ${DEVICE} \
		--num_workers ${NUM_WORKERS} \
		--split "train" \
	cli bookcorpus default-dataset \
		--path data/bookcorpus/tokenizer.json \
		--batch_size 100 \
		--drop_last False \
		--size ${SIZE} \
		--device ${DEVICE} \
		--num_workers ${NUM_WORKERS} \
		--split "validation" \
	cli bookcorpus default-dataset \
		--path data/bookcorpus/tokenizer.json \
		--batch_size 100 \
		--drop_last False \
		--size ${SIZE} \
		--device ${DEVICE} \
		--num_workers 0 \
		--split "test" \
	cli bookcorpus default-model-wt \
		--size ${SIZE} \
		--heads ${HEADS} \
		--layers ${LAYERS} \
		--dropout ${DROPOUT} \
		--embedding_size ${EMBEDDING_SIZE} \
		--feedforward_size ${FEEDFORWARD_SIZE} \
		--activation ${ACTIVATION} \
		--src_vocab_size $(shell expr ${VOCABSIZE} + 110) \
		--device ${DEVICE} \
	cli optimizers adam --learning_rate ${LR} \
	cli schedulers noscheduler \
	cli savers default-saver \
		--dirpath $(dir $@) \
		--restorepath ${RESTOREPATH} \
		--map_location ${DEVICE} \
		--savelast False \
		--savebest True \
		--mode "max" \
		--etc ${ETC} \
	cli train \
		--mini_steps 4

data/bookcorpus/seed42/test_epoch.log: data/bookcorpus/seed42/modellast.pt
	mkdir -p $(dir $@)
	PYTHONPATH=:.:semeqv \
	python3 semeqv/cli.py \
		--compile ${COMPILE} \
		--testbar True \
	cli callbacks log testcallback \
		--step_log_path  $(dir $@)test_step.log \
		--epoch_log_path $(dir $@)test_epoch.log \
	cli loss cross-entropy \
	cli bookcorpus default-dataset \
		--path data/bookcorpus/tokenizer.json \
		--batch_size ${TRAIN_BATCH_SIZE} \
		--drop_last True \
		--size ${SIZE} \
		--device ${DEVICE} \
		--num_workers 0 \
		--split "train" \
	cli bookcorpus default-dataset \
		--path data/bookcorpus/tokenizer.json \
		--batch_size 100 \
		--drop_last False \
		--size ${SIZE} \
		--device ${DEVICE} \
		--num_workers 0 \
		--split "validation" \
	cli bookcorpus default-dataset \
		--path data/bookcorpus/tokenizer.json \
		--batch_size 100 \
		--drop_last False \
		--size ${SIZE} \
		--device ${DEVICE} \
		--num_workers ${NUM_WORKERS} \
		--split "test" \
	cli bookcorpus default-model-wt \
		--size ${SIZE} \
		--heads ${HEADS} \
		--layers ${LAYERS} \
		--dropout 0 \
		--embedding_size ${EMBEDDING_SIZE} \
		--feedforward_size ${FEEDFORWARD_SIZE} \
		--activation ${ACTIVATION} \
		--src_vocab_size $(shell expr ${VOCABSIZE} + 110) \
		--device ${DEVICE} \
	cli optimizers adam --learning_rate ${LR} \
	cli schedulers noscheduler \
	cli savers default-saver \
		--dirpath $(dir $@) \
		--restorepath $< \
		--map_location ${DEVICE} \
		--savelast False \
		--savebest True \
		--mode "max" \
		--etc ${ETC} \
	cli test
