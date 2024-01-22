DEVICE=cuda:0
EPOCHS=150000
WINDOW=1
CUTOFF=1
YLIM=0 4

data/xor-lstm-small/%/modellast.pt data/xor-lstm-small/%/input_cdists.npy data/xor-lstm-small/%/output_cdists.npy data/xor-lstm-small/%/valid_epoch.log:
	mkdir -p $(dir $@)
	cp $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))/MakefileLSTMSmall.mk $(dir $@)makefile
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
		--layers 2 \
		--embedding_size 4 \
		--hidden_size 8 \
		--src_vocab_size 7 \
		--tgt_vocab_size 7 \
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

data/xor-lstm-small/embeddings_wt.pdf: \
	data/xor-lstm-small/42-lstmwt/input_cdists.npy \
	data/xor-lstm-small/43-lstmwt/input_cdists.npy \
	data/xor-lstm-small/44-lstmwt/input_cdists.npy \
	data/xor-lstm-small/45-lstmwt/input_cdists.npy \
	data/xor-lstm-small/46-lstmwt/input_cdists.npy
	PYTHONPATH=:.:semeqv python3 ./semeqv/cli.py cli view \
		--title "✓ weight tying" --path $@ --etc 1000 --ylim ${YLIM} \
		--indexes 0 1 "✗ distributional hyp." \
		--indexes 2 3 "✓ distributional hyp." $^

data/xor-lstm-small/input_embeddings.pdf: \
	data/xor-lstm-small/42-lstm/input_cdists.npy \
	data/xor-lstm-small/43-lstm/input_cdists.npy \
	data/xor-lstm-small/44-lstm/input_cdists.npy \
	data/xor-lstm-small/45-lstm/input_cdists.npy \
	data/xor-lstm-small/46-lstm/input_cdists.npy
	PYTHONPATH=:.:semeqv python3 ./semeqv/cli.py cli view \
		--title "✗ weight tying (input embeddings)" --path $@ --etc 1000 --ylim ${YLIM} \
		--indexes 0 1 "✗ distributional hyp." \
		--indexes 2 3 "✓ distributional hyp." $^

data/xor-lstm-small/output_embeddings.pdf: \
	data/xor-lstm-small/42-lstm/output_cdists.npy \
	data/xor-lstm-small/43-lstm/output_cdists.npy \
	data/xor-lstm-small/44-lstm/output_cdists.npy \
	data/xor-lstm-small/45-lstm/output_cdists.npy \
	data/xor-lstm-small/46-lstm/output_cdists.npy
	PYTHONPATH=:.:semeqv python3 ./semeqv/cli.py cli view \
		--title "✗ weight tying (output embeddings)" --path $@ --etc 1000 --ylim ${YLIM} \
		--indexes 0 1 "✗ distributional hyp." \
		--indexes 2 3 "✓ distributional hyp." $^

data/xor-lstm-small/accuracies.pdf: \
	data/xor-lstm-small/42-lstm/valid_epoch.log \
	data/xor-lstm-small/42-lstmwt/valid_epoch.log \
	data/xor-lstm-small/43-lstm/valid_epoch.log \
	data/xor-lstm-small/43-lstmwt/valid_epoch.log \
	data/xor-lstm-small/44-lstm/valid_epoch.log \
	data/xor-lstm-small/44-lstmwt/valid_epoch.log \
	data/xor-lstm-small/45-lstm/valid_epoch.log \
	data/xor-lstm-small/45-lstmwt/valid_epoch.log \
	data/xor-lstm-small/46-lstm/valid_epoch.log \
	data/xor-lstm-small/46-lstmwt/valid_epoch.log
	PYTHONPATH=:.:semeqv python3 ./semeqv/cli.py cli accplot \
		--title "Accuracy" \
		--inputs $(shell python -c "print(\" --inputs \".join([p  + \" \" + (\"\\\"✓ weight tying\\\"\" if \"wt\" in p else \"\\\"✗ weight tying\\\"\") for p in \"$^\".split(\" \")]))") \
		--etc 1 \
		--window ${WINDOW} \
		--cutoff ${CUTOFF} \
		--hline 0.7 "black" "--" \
		--output $@

figs: \
	data/xor-lstm-small/accuracies.pdf \
	data/xor-lstm-small/embeddings_wt.pdf \
	data/xor-lstm-small/input_embeddings.pdf \
	data/xor-lstm-small/output_embeddings.pdf
