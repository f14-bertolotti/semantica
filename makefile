WEIGHT_TYING=False
NUM_WORKERS=1
RESTOREPATH=""
SEED=42
EPOCHS=10
DEVICE="cuda:0"
COMPILE=True
ETC=0
LR=0.0001
SIZE=6


data/xor/wt-seed%/modelbest.pt:
	mkdir -p $(dir $@)
	PYTHONPATH=:.:semeqv \
	python3 semeqv/cli.py \
		--seed $* \
		--compile ${COMPILE} \
		--epochs ${EPOCHS} \
	cli callbacks cdists traincallback \
		--etc 100 \
		--path $(dir $@)/cdists \
		--step_log_path  $(dir $@)/train_step.log \
		--epoch_log_path $(dir $@)/train_epoch.log \
	cli callbacks log validcallback \
		--step_log_path  $(dir $@)/valid_step.log \
		--epoch_log_path $(dir $@)/valid_epoch.log \
	cli loss cross-entropy \
	cli xor default-dataset trainsplit \
		--zero_dst "[.1,.9]" \
		--one_dst "[.5,.5]" \
		--drop_last False \
		--batch_size 128 \
		--size ${SIZE} \
		--num_workers ${NUM_WORKERS} \
		--device ${DEVICE} \
	cli xor default-dataset validsplit \
		--zero_dst "[.1,.9]" \
		--one_dst "[.5,.5]" \
		--drop_last False \
		--batch_size 128 \
		--size ${SIZE} \
		--num_workers ${NUM_WORKERS} \
		--device ${DEVICE} \
	cli xor default-model \
		--heads 4 \
		--layers 3 \
		--embedding_size 128 \
		--feedforward_size 512 \
		--src_vocab_size 7 \
		--tgt_vocab_size 7 \
		--semeqvinit 2 2 \
		--weight_tying True \
		--device ${DEVICE} \
	cli optimizers adam --learning_rate ${LR} \
	cli savers default-saver \
		--dirpath $(dir $@) \
		--restorepath ${RESTOREPATH} \
		--map_location ${DEVICE} \
		--savelast False \
		--savebest True \
		--mode "max" \
		--etc ${ETC} \
	cli train

clean-xor:
	rm -rf data/xor
	

