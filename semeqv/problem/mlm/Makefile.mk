DEVICE=cuda:0
EPOCHS=3
WINDOW=1
CUTOFF=0
YLIM=0 5

data/mlm-transformer/%/modellast.pt data/mlm-transformer/%/input_cdists.npy data/mlm-transformer/%/output_cdists.npy data/mlm-transformer/%/valid_epoch.log:
	mkdir -p $(dir $@)
	cp $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))/Makefile.mk $(dir $@)makefile
	PYTHONPATH=:.:semeqv \
	python3 semeqv/cli.py \
		--seed $(shell python -c "print(\"$*\".split(\"-\")[0])") \
		--compile False \
		--epochs ${EPOCHS} \
		--trainbar True \
		--validbar True \
		--epochbar False \
		--etv 1 \
	cli mlm dataset default \
		--drop_last False \
		--batch_size 128 \
		--map_batch_size 10000 \
		--max_length 128 \
		--shuffle True \
		--device ${DEVICE} \
		--split "train" \
	cli mlm dataset default \
		--drop_last False \
		--batch_size 128 \
		--map_batch_size 10000 \
		--max_length 128 \
		--shuffle False \
		--device ${DEVICE} \
		--split "validation" \
	cli mlm callbacks cdists \
		--etc 1 \
		--stc 10000 \
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
	cli mlm model $(shell python -c "print(\"$*\".split(\"-\")[1])") \
		--heads 4 \
		--layers 3 \
		--embedding_size 128 \
		--feedforward_size 512 \
		--activation "gelu" \
		--dropout 0 \
		--device ${DEVICE} \
	cli optimizers adamw --learning_rate 0.0001 --weight_decay 0.01 \
	cli schedulers cosine --lrmin 0.00001 --tmax 10000 \
	cli savers default-saver \
		--dirpath $(dir $@) \
		--map_location ${DEVICE} \
		--savelast False \
		--savebest True \
		--mode "max" \
		--etc 0 \
	cli train

data/mlm-transformer/embeddings_wt.pdf: \
	data/mlm-transformer/42-transformerwt/input_cdists.npy
	PYTHONPATH=:.:semeqv python3 ./semeqv/cli.py cli mlm view \
		--title "✓ weight tying" \
		--path $@ \
		--etc 1 \
		--ylim ${YLIM} \
		$^

data/mlm-transformer/input_embeddings.pdf: \
	data/mlm-transformer/42-transformer/input_cdists.npy
	PYTHONPATH=:.:semeqv python3 ./semeqv/cli.py cli mlm view \
		--title "✗ weight tying (input embeddings)" --path $@ --etc 1 --ylim ${YLIM} $^

data/mlm-transformer/output_embeddings.pdf: \
	data/mlm-transformer/42-transformer/output_cdists.npy
	PYTHONPATH=:.:semeqv python3 ./semeqv/cli.py cli mlm view \
		--title "✗ weight tying (output embeddings)" --path $@ --etc 1 --ylim ${YLIM} $^

figs: \
	data/mlm-transformer/embeddings_wt.pdf \
	data/mlm-transformer/input_embeddings.pdf \
	data/mlm-transformer/output_embeddings.pdf
