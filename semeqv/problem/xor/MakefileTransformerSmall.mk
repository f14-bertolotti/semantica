DEVICE=cuda:0
EPOCHS=150000
WINDOW=1
CUTOFF=1
YLIM=0 4

data/xor-transformer-small/%/modellast.pt data/xor-transformer-small/%/input_cdists.npy data/xor-transformer-small/%/output_cdists.npy data/xor-transformer-small/%/valid_epoch.log:
	mkdir -p $(dir $@)
	cp $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))/MakefileTransformerSmall.mk $(dir $@)makefile
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
	cli train 

data/xor-transformer-small/embeddings_wt.pdf: \
	data/xor-transformer-small/42-transformerwt/input_cdists.npy \
	data/xor-transformer-small/43-transformerwt/input_cdists.npy \
	data/xor-transformer-small/44-transformerwt/input_cdists.npy \
	data/xor-transformer-small/45-transformerwt/input_cdists.npy \
	data/xor-transformer-small/46-transformerwt/input_cdists.npy
	PYTHONPATH=:.:semeqv python3 ./semeqv/cli.py cli view \
		--title "✓ weight tying" --path $@ --etc 1000 --ylim ${YLIM} \
		--indexes 0 1 "✗ distributional hyp." \
		--indexes 2 3 "✓ distributional hyp." $^

data/xor-transformer-small/input_embeddings.pdf: \
	data/xor-transformer-small/42-transformer/input_cdists.npy \
	data/xor-transformer-small/43-transformer/input_cdists.npy \
	data/xor-transformer-small/44-transformer/input_cdists.npy \
	data/xor-transformer-small/45-transformer/input_cdists.npy \
	data/xor-transformer-small/46-transformer/input_cdists.npy
	PYTHONPATH=:.:semeqv python3 ./semeqv/cli.py cli view \
		--title "✗ weight tying (input embeddings)" --path $@ --etc 1000 --ylim ${YLIM} \
		--indexes 0 1 "✗ distributional hyp." \
		--indexes 2 3 "✓ distributional hyp." $^

data/xor-transformer-small/output_embeddings.pdf: \
	data/xor-transformer-small/42-transformer/output_cdists.npy \
	data/xor-transformer-small/43-transformer/output_cdists.npy \
	data/xor-transformer-small/44-transformer/output_cdists.npy \
	data/xor-transformer-small/45-transformer/output_cdists.npy \
	data/xor-transformer-small/46-transformer/output_cdists.npy
	PYTHONPATH=:.:semeqv python3 ./semeqv/cli.py cli view \
		--title "✗ weight tying (output embeddings)" --path $@ --etc 1000 --ylim ${YLIM} \
		--indexes 0 1 "✗ distributional hyp." \
		--indexes 2 3 "✓ distributional hyp." $^

data/xor-transformer-small/accuracies.pdf: \
	data/xor-transformer-small/42-transformer/valid_epoch.log \
	data/xor-transformer-small/42-transformerwt/valid_epoch.log \
	data/xor-transformer-small/43-transformer/valid_epoch.log \
	data/xor-transformer-small/43-transformerwt/valid_epoch.log \
	data/xor-transformer-small/44-transformer/valid_epoch.log \
	data/xor-transformer-small/44-transformerwt/valid_epoch.log \
	data/xor-transformer-small/45-transformer/valid_epoch.log \
	data/xor-transformer-small/45-transformerwt/valid_epoch.log \
	data/xor-transformer-small/46-transformer/valid_epoch.log \
	data/xor-transformer-small/46-transformerwt/valid_epoch.log
	PYTHONPATH=:.:semeqv python3 ./semeqv/cli.py cli accplot \
		--title "Accuracy" \
		--inputs $(shell python -c "print(\" --inputs \".join([p  + \" \" + (\"\\\"✓ weight tying\\\"\" if \"wt\" in p else \"\\\"✗ weight tying\\\"\") for p in \"$^\".split(\" \")]))") \
		--etc 1 \
		--window ${WINDOW} \
		--cutoff ${CUTOFF} \
		--hline 0.7 "black" "--" \
		--output $@

figs: \
	data/xor-transformer-small/accuracies.pdf \
	data/xor-transformer-small/embeddings_wt.pdf \
	data/xor-transformer-small/input_embeddings.pdf \
	data/xor-transformer-small/output_embeddings.pdf
