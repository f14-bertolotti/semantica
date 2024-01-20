DEVICE=cuda:0

data/xor-ydh/%/modellast.pt data/xor-ydh/%/input_cdists.npy data/xor-ydh/%/output_cdists.npy:
	mkdir -p $(dir $@)
	cp $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))/MakefileSmallYDH.mk $(dir $@)makefile
	PYTHONPATH=:.:semeqv \
	python3 semeqv/cli.py \
		--seed $(shell python -c "print(\"$*\".split(\"-\")[0])") \
		--compile False \
		--epochs 150000 \
		--trainbar False \
		--validbar False \
		--epochbar True \
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
		--zero_dst [1,0] \
		--one_dst  [.5,.5] \
		--drop_last False \
		--batch_size 1024 \
		--shuffle True \
		--size 7 \
		--device ${DEVICE} \
		--split "train" \
	cli xor dataset default \
		--zero_dst [1,0] \
		--one_dst  [.5,.5] \
		--drop_last False \
		--batch_size 1024 \
		--shuffle False \
		--size 7 \
		--device ${DEVICE} \
		--split "validation" \
	cli xor model $(shell python -c "print(\"$*\".split(\"-\")[1])") \
		--heads 1 \
		--layers 1 \
		--embedding_size 4 \
		--feedforward_size 4 \
		--activation "gelu" \
		--src_vocab_size 7 \
		--tgt_vocab_size 7 \
		--dropout 0 \
		--device ${DEVICE} \
	cli optimizers adamw --learning_rate 0.0005 --weight_decay 0.1 \
	cli schedulers cosine --lrmin 0.00001 --tmax 10000 \
	cli savers default-saver \
		--dirpath $(dir $@) \
		--map_location ${DEVICE} \
		--savelast False \
		--savebest True \
		--mode "max" \
		--etc 0 \
	cli train --amp False --etv 100

data/xor-ydh/embeddings_wt.pdf: \
	data/xor-ydh/42-transformerwt/input_cdists.npy \
	data/xor-ydh/43-transformerwt/input_cdists.npy \
	data/xor-ydh/44-transformerwt/input_cdists.npy \
	data/xor-ydh/45-transformerwt/input_cdists.npy \
	data/xor-ydh/46-transformerwt/input_cdists.npy
	PYTHONPATH=:.:semeqv python3 ./semeqv/cli.py cli view \
		--title "✓ weight tying" --path $@ --etc 1000 \
		--indexes 2 3 "✓ distributional hyp." $^

data/xor-ydh/input_embeddings.pdf: \
	data/xor-ydh/42-transformer/input_cdists.npy \
	data/xor-ydh/43-transformer/input_cdists.npy \
	data/xor-ydh/44-transformer/input_cdists.npy \
	data/xor-ydh/45-transformer/input_cdists.npy \
	data/xor-ydh/46-transformer/input_cdists.npy
	PYTHONPATH=:.:semeqv python3 ./semeqv/cli.py cli view \
		--title "✗ weight tying (input embeddings)" --path $@ --etc 1000 \
		--indexes 2 3 "✓ distributional hyp." $^

data/xor-ydh/output_embeddings.pdf: \
	data/xor-ydh/42-transformer/output_cdists.npy \
	data/xor-ydh/43-transformer/output_cdists.npy \
	data/xor-ydh/44-transformer/output_cdists.npy \
	data/xor-ydh/45-transformer/output_cdists.npy \
	data/xor-ydh/46-transformer/output_cdists.npy
	PYTHONPATH=:.:semeqv python3 ./semeqv/cli.py cli view \
		--title "✗ weight tying (output embeddings)" --path $@ --etc 1000 \
		--indexes 2 3 "✓ distributional hyp." $^

data/xor-ydh/accuracies.pdf: \
	data/xor-ndh/42-transformer/valid_epoch.log \
	data/xor-ndh/42-transformerwt/valid_epoch.log \
	data/xor-ndh/43-transformer/valid_epoch.log \
	data/xor-ndh/43-transformerwt/valid_epoch.log \
	data/xor-ndh/44-transformer/valid_epoch.log \
	data/xor-ndh/44-transformerwt/valid_epoch.log \
	data/xor-ndh/45-transformer/valid_epoch.log \
	data/xor-ndh/45-transformerwt/valid_epoch.log \
	data/xor-ndh/46-transformer/valid_epoch.log \
	data/xor-ndh/46-transformerwt/valid_epoch.log
	PYTHONPATH=:.:semeqv python3 ./semeqv/cli.py cli accplot \
		--title "Accuracy" \
		--inputs $(shell python -c "print(\" --inputs \".join([p  + \" \" + (\"\\\"✓ weight tying\\\"\" if \"wt\" in p else \"\\\"✗ weight tying\\\"\") for p in \"$^\".split(\" \")]))") \
		--etc 1 \
		--window 1 \
		--hline 0.7 "black" "--" \
		--output $@

figs: \
	data/xor-ydh/accuracies.pdf \
	data/xor-ydh/embeddings_wt.pdf \
	data/xor-ydh/input_embeddings.pdf \
	data/xor-ydh/output_embeddings.pdf
