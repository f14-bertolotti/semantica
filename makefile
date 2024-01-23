WINDOW=1
EPOCHSMALL=150000
EPOCHMED=500000
CUTOFF=1
DEVICE=cuda:0

MLM_EPOCHS=3
MLM_CUTOFF=0
MLM_WINDOW=1

mlm:
	CUTOFF=${MLM_CUTOFF} WINDOW=${MLM_WINDOW} EPOCHS=${MLM_EPOCHS}   make -e -f semeqv/problem/mlm/Makefile.mk figs

xor:
	CUTOFF=${CUTOFF} WINDOW=${WINDOW} EPOCHS=${EPOCHSMALL} make -e -f semeqv/problem/xor/MakefileTransformerSmall.mk figs
	CUTOFF=${CUTOFF} WINDOW=${WINDOW} EPOCHS=${EPOCHMED}   make -e -f semeqv/problem/xor/MakefileLSTMMedium.mk figs
	CUTOFF=${CUTOFF} WINDOW=${WINDOW} EPOCHS=${EPOCHSMALL} make -e -f semeqv/problem/xor/MakefileLSTMSmall.mk figs
	CUTOFF=${CUTOFF} WINDOW=${WINDOW} EPOCHS=${EPOCHMED}   make -e -f semeqv/problem/xor/MakefileMLPMedium.mk figs
	CUTOFF=${CUTOFF} WINDOW=${WINDOW} EPOCHS=${EPOCHSMALL} make -e -f semeqv/problem/xor/MakefileMLPSmall.mk figs
	CUTOFF=${CUTOFF} WINDOW=${WINDOW} EPOCHS=${EPOCHMED}   make -e -f semeqv/problem/xor/MakefileTransformerMedium.mk figs
	CUTOFF=${CUTOFF} WINDOW=${WINDOW} EPOCHS=${EPOCHSMALL} make -e -f semeqv/problem/xor/MakefileTransformerSmall.mk figs
	CUTOFF=${CUTOFF} WINDOW=${WINDOW} EPOCHS=${EPOCHSMALL} make -e -f semeqv/problem/xor/MakefileTransformerSmallNDH.mk figs
	CUTOFF=${CUTOFF} WINDOW=${WINDOW} EPOCHS=${EPOCHSMALL} make -e -f semeqv/problem/xor/MakefileTransformerSmallYDH.mk figs

clean:
	find ./semeqv/ -name "*.pyc" -exec rm -f {} \;
	find ./semeqv/ -name "__pycache__" -exec rm -rf {} \;
	rm -rf venv

docker-build:
	docker build ./semeqv -t f14:semeqv -f ./Dockerfile

docker-run:
	docker run --gpus all --rm \
		--user="$(shell id -u)" \
		--mount type=bind,src="${PWD}",dst="${PWD}" \
		--env PYTHONPATH=/semeqv/semeqv/:/semeqv/ \
		-w "${PWD}" -it f14:semeqv bash
