WEIGHT_TYING=False
NUM_WORKERS=0
RESTOREPATH=""
SEED=42
EPOCHS=500000
DEVICE="cuda:0"
COMPILE=False
ETC=0
ETD=250000
LR=0.001
LRMIN=0.0001
TMAX=1000
SIZE=7

EMBEDDING_SIZE=16
FEEDFORWARD_SIZE=64
LAYERS=3
HEADS=2


data/xor/seed%-wt-init/modellast.pt:
	mkdir -p $(dir $@)
	PYTHONPATH=:.:semeqv \
	python3 semeqv/cli.py \
		--seed $* \
		--compile ${COMPILE} \
		--epochs ${EPOCHS} \
		--trainbar False \
		--validbar False \
		--epochbar True \
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
		--batch_size 1024 \
		--size ${SIZE} \
		--num_workers ${NUM_WORKERS} \
		--device ${DEVICE} \
	cli xor default-dataset validsplit \
		--zero_dst "[.1,.9]" \
		--one_dst "[.5,.5]" \
		--drop_last False \
		--batch_size 1024 \
		--size ${SIZE} \
		--num_workers ${NUM_WORKERS} \
		--device ${DEVICE} \
	cli xor default-model \
		--heads ${HEADS} \
		--layers ${LAYERS} \
		--embedding_size ${EMBEDDING_SIZE} \
		--feedforward_size ${FEEDFORWARD_SIZE} \
		--activation "relu" \
		--src_vocab_size 7 \
		--tgt_vocab_size 7 \
		--semeqvinit 2 2 \
		--weight_tying True \
		--device ${DEVICE} \
	cli optimizers adam --learning_rate ${LR} \
	cli schedulers cosine --lrmin ${LRMIN} --tmax ${TMAX} \
	cli savers default-saver \
		--dirpath $(dir $@) \
		--restorepath ${RESTOREPATH} \
		--map_location ${DEVICE} \
		--savelast False \
		--savebest True \
		--mode "max" \
		--etc ${ETC} \
	cli train

data/xor/seed%-init/modellast.pt:
	mkdir -p $(dir $@)
	PYTHONPATH=:.:semeqv \
	python3 semeqv/cli.py \
		--seed $* \
		--compile ${COMPILE} \
		--epochs ${EPOCHS} \
		--trainbar False \
		--validbar False \
		--epochbar True \
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
		--batch_size 1024 \
		--size ${SIZE} \
		--num_workers ${NUM_WORKERS} \
		--device ${DEVICE} \
	cli xor default-dataset validsplit \
		--zero_dst "[.1,.9]" \
		--one_dst "[.5,.5]" \
		--drop_last False \
		--batch_size 1024 \
		--size ${SIZE} \
		--num_workers ${NUM_WORKERS} \
		--device ${DEVICE} \
	cli xor default-model \
		--heads ${HEADS} \
		--layers ${LAYERS} \
		--embedding_size ${EMBEDDING_SIZE} \
		--feedforward_size ${FEEDFORWARD_SIZE} \
		--activation "relu" \
		--src_vocab_size 7 \
		--tgt_vocab_size 7 \
		--semeqvinit 2 2 \
		--weight_tying False \
		--device ${DEVICE} \
	cli optimizers adam --learning_rate ${LR} \
	cli schedulers cosine --lrmin ${LRMIN} --tmax ${TMAX} \
	cli savers default-saver \
		--dirpath $(dir $@) \
		--restorepath ${RESTOREPATH} \
		--map_location ${DEVICE} \
		--savelast False \
		--savebest True \
		--mode "max" \
		--etc ${ETC} \
	cli train

data/xor/seed%-wt/modellast.pt:
	mkdir -p $(dir $@)
	PYTHONPATH=:.:semeqv \
	python3 semeqv/cli.py \
		--seed $* \
		--compile ${COMPILE} \
		--epochs ${EPOCHS} \
		--trainbar False \
		--validbar False \
		--epochbar True \
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
		--batch_size 1024 \
		--size ${SIZE} \
		--num_workers ${NUM_WORKERS} \
		--device ${DEVICE} \
	cli xor default-dataset validsplit \
		--zero_dst "[.1,.9]" \
		--one_dst "[.5,.5]" \
		--drop_last False \
		--batch_size 1024 \
		--size ${SIZE} \
		--num_workers ${NUM_WORKERS} \
		--device ${DEVICE} \
	cli xor default-model \
		--heads ${HEADS} \
		--layers ${LAYERS} \
		--embedding_size ${EMBEDDING_SIZE} \
		--feedforward_size ${FEEDFORWARD_SIZE} \
		--activation "relu" \
		--src_vocab_size 7 \
		--tgt_vocab_size 7 \
		--semeqvinit 0 0 \
		--weight_tying True \
		--device ${DEVICE} \
	cli optimizers adam --learning_rate ${LR} \
	cli schedulers cosine --lrmin ${LRMIN} --tmax ${TMAX} \
	cli savers default-saver \
		--dirpath $(dir $@) \
		--restorepath ${RESTOREPATH} \
		--map_location ${DEVICE} \
		--savelast False \
		--savebest True \
		--mode "max" \
		--etc ${ETC} \
	cli train

data/xor/seed%/modellast.pt:
	mkdir -p $(dir $@)
	PYTHONPATH=:.:semeqv \
	python3 semeqv/cli.py \
		--seed $* \
		--compile ${COMPILE} \
		--epochs ${EPOCHS} \
		--trainbar False \
		--validbar False \
		--epochbar True \
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
		--batch_size 1024 \
		--size ${SIZE} \
		--num_workers ${NUM_WORKERS} \
		--device ${DEVICE} \
	cli xor default-dataset validsplit \
		--zero_dst "[.1,.9]" \
		--one_dst "[.5,.5]" \
		--drop_last False \
		--batch_size 1024 \
		--size ${SIZE} \
		--num_workers ${NUM_WORKERS} \
		--device ${DEVICE} \
	cli xor default-model \
		--heads ${HEADS} \
		--layers ${LAYERS} \
		--embedding_size ${EMBEDDING_SIZE} \
		--feedforward_size ${FEEDFORWARD_SIZE} \
		--activation "relu" \
		--src_vocab_size 7 \
		--tgt_vocab_size 7 \
		--semeqvinit 0 0 \
		--weight_tying False \
		--device ${DEVICE} \
	cli optimizers adam --learning_rate ${LR} \
	cli schedulers cosine --lrmin ${LRMIN} --tmax ${TMAX} \
	cli savers default-saver \
		--dirpath $(dir $@) \
		--restorepath ${RESTOREPATH} \
		--map_location ${DEVICE} \
		--savelast False \
		--savebest True \
		--mode "max" \
		--etc ${ETC} \
	cli train

data/xor/seed%-decouple/modellast.pt:
	mkdir -p $(dir $@)
	PYTHONPATH=:.:semeqv \
	python3 semeqv/cli.py \
		--seed $* \
		--compile ${COMPILE} \
		--epochs ${EPOCHS} \
		--trainbar False \
		--validbar False \
		--epochbar True \
	cli callbacks cdists-decouple traincallback \
		--etc 100 \
		--etd ${ETD} \
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
		--batch_size 1024 \
		--size ${SIZE} \
		--num_workers ${NUM_WORKERS} \
		--device ${DEVICE} \
	cli xor default-dataset validsplit \
		--zero_dst "[.1,.9]" \
		--one_dst "[.5,.5]" \
		--drop_last False \
		--batch_size 1024 \
		--size ${SIZE} \
		--num_workers ${NUM_WORKERS} \
		--device ${DEVICE} \
	cli xor default-model \
		--heads ${HEADS} \
		--layers ${LAYERS} \
		--embedding_size ${EMBEDDING_SIZE} \
		--feedforward_size ${FEEDFORWARD_SIZE} \
		--activation "relu" \
		--src_vocab_size 7 \
		--tgt_vocab_size 7 \
		--semeqvinit 0 0 \
		--weight_tying True \
		--device ${DEVICE} \
	cli optimizers adam --learning_rate ${LR} \
	cli schedulers cosine --lrmin ${LRMIN} --tmax ${TMAX} \
	cli savers default-saver \
		--dirpath $(dir $@) \
		--restorepath ${RESTOREPATH} \
		--map_location ${DEVICE} \
		--savelast False \
		--savebest True \
		--mode "max" \
		--etc ${ETC} \
	cli train



xor-wt-init: \
	data/xor/seed42-wt-init/modellast.pt \
	data/xor/seed43-wt-init/modellast.pt \
	data/xor/seed44-wt-init/modellast.pt \
	data/xor/seed45-wt-init/modellast.pt \
	data/xor/seed46-wt-init/modellast.pt

xor-init: \
	data/xor/seed42-init/modellast.pt \
	data/xor/seed43-init/modellast.pt \
	data/xor/seed44-init/modellast.pt \
	data/xor/seed45-init/modellast.pt \
	data/xor/seed46-init/modellast.pt

xor-wt: \
	data/xor/seed42-wt/modellast.pt \
	data/xor/seed43-wt/modellast.pt \
	data/xor/seed44-wt/modellast.pt \
	data/xor/seed45-wt/modellast.pt \
	data/xor/seed46-wt/modellast.pt

xor: \
	data/xor/seed42/modellast.pt \
	data/xor/seed43/modellast.pt \
	data/xor/seed44/modellast.pt \
	data/xor/seed45/modellast.pt \
	data/xor/seed46/modellast.pt

data/xor/xor.pdf: \
	data/xor/seed42/modellast.pt \
	data/xor/seed43/modellast.pt \
	data/xor/seed44/modellast.pt \
	data/xor/seed45/modellast.pt \
	data/xor/seed46/modellast.pt
	PYTHONPATH=:.:semeqv python3 ./semeqv/cli.py cli view \
		--title "✗ semeqv init, ✗ weight tying" \
		--path $@ \
		--etc 300 \
		--indexes 0 1 \
		--indexes 2 3 \
		data/xor/seed42/cdists.npy data/xor/seed43/cdists.npy data/xor/seed44/cdists.npy data/xor/seed45/cdists.npy  data/xor/seed46/cdists.npy

data/xor/xor-init.pdf: \
	data/xor/seed42-init/modellast.pt \
	data/xor/seed43-init/modellast.pt \
	data/xor/seed44-init/modellast.pt \
	data/xor/seed45-init/modellast.pt \
	data/xor/seed46-init/modellast.pt
	PYTHONPATH=:.:semeqv python3 ./semeqv/cli.py cli view \
		--title "✓ semeqv init, ✗ weight tying" \
		--path $@ \
		--etc 300 \
		--indexes 0 1 \
		--indexes 2 3 \
		data/xor/seed42-init/cdists.npy data/xor/seed43-init/cdists.npy data/xor/seed44-init/cdists.npy data/xor/seed45-init/cdists.npy data/xor/seed46-init/cdists.npy

data/xor/xor-wt.pdf: \
	data/xor/seed42-wt/modellast.pt \
	data/xor/seed43-wt/modellast.pt \
	data/xor/seed44-wt/modellast.pt \
	data/xor/seed45-wt/modellast.pt \
	data/xor/seed46-wt/modellast.pt
	PYTHONPATH=:.:semeqv python3 ./semeqv/cli.py cli view \
		--title "✗ semeqv init, ✓ weight tying" \
		--path $@ \
		--etc 300 \
		--indexes 0 1 \
		--indexes 2 3 \
		data/xor/seed42-wt/cdists.npy data/xor/seed43-wt/cdists.npy data/xor/seed44-wt/cdists.npy data/xor/seed45-wt/cdists.npy data/xor/seed46-wt/cdists.npy

data/xor/xor-wt-init.pdf: \
	data/xor/seed42-wt-init/modellast.pt \
	data/xor/seed43-wt-init/modellast.pt \
	data/xor/seed44-wt-init/modellast.pt \
	data/xor/seed45-wt-init/modellast.pt \
	data/xor/seed46-wt-init/modellast.pt
	PYTHONPATH=:.:semeqv python3 ./semeqv/cli.py cli view \
		--title "✓ semeqv init, ✓ weight tying" \
		--path $@ \
		--etc 300 \
		--indexes 0 1 \
		--indexes 2 3 \
		data/xor/seed42-wt-init/cdists.npy data/xor/seed43-wt-init/cdists.npy data/xor/seed44-wt-init/cdists.npy data/xor/seed45-wt-init/cdists.npy data/xor/seed46-wt-init/cdists.npy

pdfs: \
	data/xor/xor.pdf \
	data/xor/xor-init.pdf \
	data/xor/xor-wt.pdf \
	data/xor/xor-wt-init.pdf

clean-xor:
	rm -rf data/xor
	

