DEVICE=cuda:0
EPOCHS=300000

data/xor-transformer-medium/%/modellast.pt data/xor-transformer-medium/%/input_cdists.npy data/xor-transformer-medium/%/output_cdists.npy data/xor-transformer-medium/%/valid_epoch.log:
	mkdir -p $(dir $@)
	cp $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))/MakefileTransformerMedium.mk $(dir $@)makefile
	PYTHONPATH=:.:semeqv \
	python3 semeqv/cli.py \
		--seed $(shell python -c "print(\"$*\".split(\"-\")[0])") \
		--compile False \
		--epochs ${EPOCHS} \
		--trainbar False \
		--validbar False \
		--epochbar True \
		--etv 100 \
	cli callbacks cdists \
		--etc 10000 \
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
		--shuffle True \
		--size 7 \
		--device ${DEVICE} \
		--split "train" \
	cli xor dataset default \
		--zero_dst [.1,.9] \
		--one_dst  [.5,.5] \
		--drop_last False \
		--batch_size 1024 \
		--shuffle False \
		--size 7 \
		--device ${DEVICE} \
		--split "validation" \
	cli xor model $(shell python -c "print(\"$*\".split(\"-\")[1])") \
		--heads 1 \
		--layers 2 \
		--embedding_size 32 \
		--feedforward_size 32 \
		--activation "gelu" \
		--src_vocab_size 7 \
		--tgt_vocab_size 7 \
		--dropout 0 \
		--device ${DEVICE} \
	cli optimizers adamw --learning_rate 0.0001 --weight_decay 0.1 \
	cli schedulers cosine --lrmin 0.00001 --tmax 10000 \
	cli savers default-saver \
		--dirpath $(dir $@) \
		--map_location ${DEVICE} \
		--savelast False \
		--savebest True \
		--mode "max" \
		--etc 0 \
	cli train

data/xor-transformer-medium/embeddings_wt.pdf: \
	data/xor-transformer-medium/42-transformerwt/input_cdists.npy \
	data/xor-transformer-medium/43-transformerwt/input_cdists.npy \
	data/xor-transformer-medium/44-transformerwt/input_cdists.npy \
	data/xor-transformer-medium/45-transformerwt/input_cdists.npy \
	data/xor-transformer-medium/46-transformerwt/input_cdists.npy
	PYTHONPATH=:.:semeqv python3 ./semeqv/cli.py cli view \
		--title "✓ weight tying" --path $@ --etc 1000 \
		--indexes 0 1 "✗ distributional hyp." \
		--indexes 2 3 "✓ distributional hyp." $^

data/xor-transformer-medium/input_embeddings.pdf: \
	data/xor-transformer-medium/42-transformer/input_cdists.npy \
	data/xor-transformer-medium/43-transformer/input_cdists.npy \
	data/xor-transformer-medium/44-transformer/input_cdists.npy \
	data/xor-transformer-medium/45-transformer/input_cdists.npy \
	data/xor-transformer-medium/46-transformer/input_cdists.npy
	PYTHONPATH=:.:semeqv python3 ./semeqv/cli.py cli view \
		--title "✗ weight tying (input embeddings)" --path $@ --etc 1000 \
		--indexes 0 1 "✗ distributional hyp." \
		--indexes 2 3 "✓ distributional hyp." $^

data/xor-transformer-medium/output_embeddings.pdf: \
	data/xor-transformer-medium/42-transformer/output_cdists.npy \
	data/xor-transformer-medium/43-transformer/output_cdists.npy \
	data/xor-transformer-medium/44-transformer/output_cdists.npy \
	data/xor-transformer-medium/45-transformer/output_cdists.npy \
	data/xor-transformer-medium/46-transformer/output_cdists.npy
	PYTHONPATH=:.:semeqv python3 ./semeqv/cli.py cli view \
		--title "✗ weight tying (output embeddings)" --path $@ --etc 1000 \
		--indexes 0 1 "✗ distributional hyp." \
		--indexes 2 3 "✓ distributional hyp." $^

data/xor-transformer-medium/accuracies.pdf: \
	data/xor-transformer-medium/42-transformer/valid_epoch.log \
	data/xor-transformer-medium/42-transformerwt/valid_epoch.log \
	data/xor-transformer-medium/43-transformer/valid_epoch.log \
	data/xor-transformer-medium/43-transformerwt/valid_epoch.log \
	data/xor-transformer-medium/44-transformer/valid_epoch.log \
	data/xor-transformer-medium/44-transformerwt/valid_epoch.log \
	data/xor-transformer-medium/45-transformer/valid_epoch.log \
	data/xor-transformer-medium/45-transformerwt/valid_epoch.log \
	data/xor-transformer-medium/46-transformer/valid_epoch.log \
	data/xor-transformer-medium/46-transformerwt/valid_epoch.log
	PYTHONPATH=:.:semeqv python3 ./semeqv/cli.py cli accplot \
		--title "Accuracy" \
		--inputs $(shell python -c "print(\" --inputs \".join([p  + \" \" + (\"\\\"✓ weight tying\\\"\" if \"wt\" in p else \"\\\"✗ weight tying\\\"\") for p in \"$^\".split(\" \")]))") \
		--etc 1 \
		--window 1 \
		--hline 0.7 "black" "--" \
		--output $@

figs: \
	data/xor-transformer-medium/accuracies.pdf \
	data/xor-transformer-medium/embeddings_wt.pdf \
	data/xor-transformer-medium/input_embeddings.pdf \
	data/xor-transformer-medium/output_embeddings.pd
