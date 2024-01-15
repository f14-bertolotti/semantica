DEVICE=cuda:0

data/xor/%/modellast.pt data/xor/%/input_cdists.npy data/xor/%/output_cdists.npy:
	mkdir -p $(dir $@)
	PYTHONPATH=:.:semeqv \
	python3 semeqv/cli.py \
		--seed $(shell python -c "print(\"$*\".split(\"-\")[0])") \
		--compile False \
		--epochs 500000 \
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
		--zero_dst [.1,.9] \
		--one_dst  [.1,.9] \
		--drop_last False \
		--batch_size 1024 \
		--shuffle True \
		--size 7 \
		--device ${DEVICE} \
		--split "train" \
	cli xor dataset default \
		--zero_dst [.1,.9] \
		--one_dst  [.1,.9] \
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
	cli optimizers adamw --learning_rate 0.001 --weight_decay 0.01 \
	cli schedulers cosine --lrmin 0.00001 --tmax 1000 \
	cli savers default-saver \
		--dirpath $(dir $@) \
		--map_location ${DEVICE} \
		--savelast False \
		--savebest True \
		--mode "max" \
		--etc 0 \
	cli train --amp False --etv 100

data/xor/embeddings_wt.pdf: \
	data/xor/42-transformerwt/input_cdists.npy \
	data/xor/43-transformerwt/input_cdists.npy \
	data/xor/44-transformerwt/input_cdists.npy \
	data/xor/45-transformerwt/input_cdists.npy \
	data/xor/46-transformerwt/input_cdists.npy
	PYTHONPATH=:.:semeqv python3 ./semeqv/cli.py cli view \
		--title "✓ weight tying" --path $@ --etc 1000 \
		--indexes 0 1 "✗ distributional hyp." \
		--indexes 2 3 "✓ distributional hyp." $^

data/xor/input_embeddings.pdf: \
	data/xor/42-transformer/input_cdists.npy \
	data/xor/43-transformer/input_cdists.npy \
	data/xor/44-transformer/input_cdists.npy \
	data/xor/45-transformer/input_cdists.npy \
	data/xor/46-transformer/input_cdists.npy
	PYTHONPATH=:.:semeqv python3 ./semeqv/cli.py cli view \
		--title "✗ weight tying (input embeddings)" --path $@ --etc 1000 \
		--indexes 0 1 "✗ distributional hyp." \
		--indexes 2 3 "✓ distributional hyp." $^

data/xor/output_embeddings.pdf: \
	data/xor/42-transformer/output_cdists.npy \
	data/xor/43-transformer/output_cdists.npy \
	data/xor/44-transformer/output_cdists.npy \
	data/xor/45-transformer/output_cdists.npy \
	data/xor/46-transformer/output_cdists.npy
	PYTHONPATH=:.:semeqv python3 ./semeqv/cli.py cli view \
		--title "✗ weight tying (output embeddings)" --path $@ --etc 1000 \
		--indexes 0 1 "✗ distributional hyp." \
		--indexes 2 3 "✓ distributional hyp." $^

data/xor/accuracies.pdf: \
	data/xor/42-transformer/valid_epoch.log \
	data/xor/43-transformer/valid_epoch.log \
	data/xor/44-transformer/valid_epoch.log \
	data/xor/45-transformer/valid_epoch.log \
	data/xor/46-transformer/valid_epoch.log \
	data/xor/42-transformerwt/valid_epoch.log \
	data/xor/43-transformerwt/valid_epoch.log \
	data/xor/44-transformerwt/valid_epoch.log \
	data/xor/45-transformerwt/valid_epoch.log \
	data/xor/46-transformerwt/valid_epoch.log
	PYTHONPATH=:.:semeqv python3 ./semeqv/cli.py cli accplot \
		--title "Accuracy" \
		--inputs $(shell python -c "print(\" --inputs \".join([p  + \" \" + (\"\\\"✓ weight tying\\\"\" if \"wt\" in p else \"\\\"✗ weight tying\\\"\") for p in \"$^\".split(\" \")]))") \
		--etc 10 \
		--window 10 \
		--hline 0.7375 "black" "--" \
		--output $@

figs: \
	data/xor/embeddings_wt.pdf \
	data/xor/input_embeddings.pdf \
	data/xor/output_embeddings.pdf \
	data/xor/accuracies.pdf
