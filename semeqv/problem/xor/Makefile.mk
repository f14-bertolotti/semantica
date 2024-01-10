DEVICE=cuda:0
COMPILE=True

data/xor/seed42/modellast.pt:
	mkdir -p $(dir $@)
	PYTHONPATH=:.:semeqv \
	python3 semeqv/cli.py \
		--seed 42 \
		--compile ${COMPILE} \
		--epochs 100000 \
		--trainbar False \
		--validbar False \
		--epochbar True \
	cli callbacks cdists \
		--etc 100 \
		--input_embedding_path  $(dir $@)/input_cdists \
		--output_embedding_path $(dir $@)/output_cdists \
		--step_log_path  $(dir $@)/train_step.log \
		--epoch_log_path $(dir $@)/train_epoch.log \
		--split "train" \
	cli callbacks log \
		--step_log_path  $(dir $@)/valid_step.log \
		--epoch_log_path $(dir $@)/valid_epoch.log \
		--split "validation" \
	cli loss cross-entropy \
	cli xor dataset default \
		--zero_dst [.1,.9] \
		--one_dst  [.5,.5] \
		--drop_last False \
		--batch_size 1024 \
		--size 5 \
		--num_workers 1 \
		--device ${DEVICE} \
		--split "train" \
	cli xor dataset default \
		--zero_dst [.1,.9] \
		--one_dst  [.5,.5] \
		--drop_last False \
		--batch_size 1024 \
		--size 5 \
		--num_workers 0 \
		--device ${DEVICE} \
		--split "validation" \
	cli xor model default \
		--heads 1 \
		--layers 3 \
		--embedding_size 4 \
		--feedforward_size 16 \
		--activation "gelu" \
		--src_vocab_size 7 \
		--tgt_vocab_size 7 \
		--dropout 0 \
		--device ${DEVICE} \
	cli optimizers adam --learning_rate 0.0001 \
	cli schedulers cosine --lrmin 0.00001 --tmax 100 \
	cli savers default-saver \
		--dirpath $(dir $@) \
		--map_location ${DEVICE} \
		--savelast False \
		--savebest True \
		--mode "max" \
		--etc 0 \
	cli train --amp False

data/xor/input_embeddings.pdf: \
	data/xor/seed42/modellast.pt
	PYTHONPATH=:.:semeqv python3 ./semeqv/cli.py cli view \
		--title "semeqv" \
		--path $@ \
		--etc 10 \
		--indexes 0 1 \
		--indexes 2 3 \
		--show True \
		data/xor/seed42/input_cdists.npy

data/xor/output_embeddings.pdf: \
	data/xor/seed42/modellast.pt
	PYTHONPATH=:.:semeqv python3 ./semeqv/cli.py cli view \
		--title "semeqv" \
		--path $@ \
		--etc 10 \
		--indexes 0 1 \
		--indexes 2 3 \
		--show True \
		data/xor/seed42/output_cdists.npy




