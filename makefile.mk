WEIGHT_TYING=False
NUM_WORKERS=1
RESTOREPATH=""
DIRPATH="./"
SEED=42
EPOCHS=10
DEVICE="cuda:0"
COMPILE=True
ETC=0
LR=0.0001

test:
	PYTHONPATH=:.:semeqv \
	python3 semeqv/cli.py \
		--seed ${SEED} \
		--compile ${COMPILE} \
		--epochs ${EPOCHS} \
	cli callbacks cdists traincallback \
		--etc 10 \
		--step_log_path  "./train_step.log" \
		--epoch_log_path "./train_epoch.log" \
		--path "cdists" \
	cli callbacks log validcallback \
		--step_log_path  "./valid_step.log" \
		--epoch_log_path "./valid_epoch.log" \
	cli loss cross-entropy \
	cli xor default-dataset trainsplit \
		--zero_dst "[.5,.5]" \
		--one_dst "[1]" \
		--drop_last False \
		--batch_size 128 \
		--size 12 \
		--num_workers ${NUM_WORKERS} \
		--device ${DEVICE} \
	cli xor default-dataset validsplit \
		--zero_dst "[.5,.5]" \
		--one_dst "[1]" \
		--drop_last False \
		--batch_size 128 \
		--size 12 \
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
		--weight_tying ${WEIGHT_TYING} \
		--device ${DEVICE} \
	cli optimizers adam --learning_rate ${LR} \
	cli savers default-saver \
		--dirpath ${DIRPATH} \
		--restorepath ${RESTOREPATH} \
		--map_location ${DEVICE} \
		--savelast False \
		--savebest False \
		--mode "max" \
		--etc ${ETC} \
	cli train

