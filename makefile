WINDOW=1
EPOCHSMALL=150000
EPOCHMED=500000
CUTOFF=1

all:
	CUTOFF=${CUTOFF} WINDOW=${WINDOW} EPOCH=${EPOCHSMAL} make -e -f semeqv/problem/xor/MakefileTransformerSmall.mk figs
	CUTOFF=${CUTOFF} WINDOW=${WINDOW} EPOCH=${EPOCHMED}  make -e -f semeqv/problem/xor/MakefileLSTMMedium.mk figs
	CUTOFF=${CUTOFF} WINDOW=${WINDOW} EPOCH=${EPOCHSMAL} make -e -f semeqv/problem/xor/MakefileLSTMSmall.mk figs
	CUTOFF=${CUTOFF} WINDOW=${WINDOW} EPOCH=${EPOCHMED}  make -e -f semeqv/problem/xor/MakefileMLPMedium.mk figs
	CUTOFF=${CUTOFF} WINDOW=${WINDOW} EPOCH=${EPOCHSMAL} make -e -f semeqv/problem/xor/MakefileMLPSmall.mk figs
	CUTOFF=${CUTOFF} WINDOW=${WINDOW} EPOCH=${EPOCHMED}  make -e -f semeqv/problem/xor/MakefileTransformerMedium.mk figs
	CUTOFF=${CUTOFF} WINDOW=${WINDOW} EPOCH=${EPOCHSMAL} make -e -f semeqv/problem/xor/MakefileTransformerSmall.mk figs
	CUTOFF=${CUTOFF} WINDOW=${WINDOW} EPOCH=${EPOCHSMAL} make -e -f semeqv/problem/xor/MakefileTransformerSmallNDH.mk figs
	CUTOFF=${CUTOFF} WINDOW=${WINDOW} EPOCH=${EPOCHSMAL} make -e -f semeqv/problem/xor/MakefileTransformerSmallYDH.mk figs

