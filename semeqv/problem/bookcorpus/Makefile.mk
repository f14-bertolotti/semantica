NUM_WORKERS=1
RESTOREPATH=""
SEED=42
EPOCHS=3
DEVICE="cuda:0"
COMPILE=False
ETC=1
LR=0.0004
SIZE=128

EMBEDDING_SIZE=128
FEEDFORWARD_SIZE=512
LAYERS=3
HEADS=4


data/bookcorpus/seed%/modellast.pt:
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
		--stc 10000 \
		--step_log_path  $(dir $@)train_step.log \
		--epoch_log_path $(dir $@)train_epoch.log \
	cli callbacks log validcallback \
		--step_log_path  $(dir $@)valid_step.log \
		--epoch_log_path $(dir $@)valid_epoch.log \
	cli loss cross-entropy \
	cli bookcorpus default-dataset trainsplit \
		--batch_size 128 \
		--drop_last True \
		--size ${SIZE} \
		--device ${DEVICE} \
		--num_workers ${NUM_WORKERS} \
	cli bookcorpus default-dataset validsplit \
		--batch_size 64 \
		--drop_last False \
		--size ${SIZE} \
		--device ${DEVICE} \
		--num_workers ${NUM_WORKERS} \
	cli bookcorpus default-model-wt \
		--size ${SIZE} \
		--heads ${HEADS} \
		--layers ${LAYERS} \
		--embedding_size ${EMBEDDING_SIZE} \
		--feedforward_size ${FEEDFORWARD_SIZE} \
		--activation "relu" \
		--src_vocab_size 30622 \
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
	cli train


