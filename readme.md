
# Semantica
This is the repository for the code relative to the ICML 2024 submission of  **By Tying Embeddings You Are Assuming the Distributional Hypothesis**.  

## Abstract
 In this work, we analyze both theoretically and empirically the effect of tied input-output embeddings---a popular technique that reduces the model size while often improving training. 
 Interestingly, we found that this technique is connected to the distributional hypothesis, often portrayed by the famous J.Firth quote *"A word is characterized by the company it keeps"*. In particular, we find that words (or, more generally, symbols) with similar semantics are encoded in similar input embeddings, while words that appear in similar contexts are encoded in similar output embeddings.
As a consequence of these findings, the tying of the input and output embeddings is encouraged only when the distributional hypothesis holds for the underlying data. These results also provide insight into the embeddings of foundation language models (which are known to be semantically organized). 
Further, we complement the theoretical findings with several experiments supporting the claims, replicable with this package.

## Requirements
To reproduce our experiments you will need to have installed `python3` with the following packages installed. We provide also a package version used at the time of writing. However, the `requirements.txt` does not specify version numbers. You will also need `make` to run makefiles. 

|package  |version |
|---------|--------|
| [click](https://click.palletsprojects.com/en/8.1.x/)           | 8.1.7    |
| [jsonlines](https://jsonlines.readthedocs.io/en/latest/)       | 4.0.01.4 |
| [matplotlib](https://matplotlib.org/)                          | 3.8.2    |
| [numpy](https://numpy.org/)                                    | 1.26.3   |
| [pandas](https://pandas.pydata.org/)                           | 2.2.0    |
| [rich](https://github.com/Textualize/rich)                     | 13.7.0   |
| [seaborn](https://seaborn.pydata.org/)                         | 0.13.1   |
| [termcolor](https://github.com/termcolor/termcolor)            | 2.4.0    |
| [torch](https://pytorch.org/)                                  | 2.1.2    |
| [tqdm](https://tqdm.github.io/)                                | 4.66.1   |
| [transformers](https://huggingface.co/docs/transformers/index) | 4.37.0   |

To install these packages, we suggest using a virtual environment, for example running:
1. `python3 -m venv venv`
2. `source venv/bin/activate`
3. `python3 -m pip install -r requirements.txt`

It should create a minimal python3 environment to run the experiments.

## Experiments
We prepared a makefile for each experiment. These makefiles can be found in the directory `semeqv/problem/xor/`:
| name   | experiment |
|--------|------------|
| `MakefileTransformerSmall.mk`    | small transformer architecture (main experiment) |
| `MakefileTransformerMedium.mk`   | larger transformer architecture                  |
| `MakefileTransformerSmallYDH.mk` | no cond.differ. symbols                          |
| `MakefileTransformerSmallNDH.mk` | no cond.eqv symbols                              |
| `MakefileLSTMSmall.mk`           | small LSTM architecture                          |
| `MakefileLSTMMedium.mk`          | larger LSTM architecture                         |
| `MakefileMLPSmall.mk`            | small MLPMixer architecture                      |
| `MakefileMLPMedium.mk`           | larger MLPMixer architecture                     |

If you desire to run all experiments, the file `makefile` provides the command `all` to run everything: `make all`.
Otherwise, to run a specific experiment you can run `make -f <path> figs`. This will train the model and plot figures as the one presented in the paper. For example, `make -f semeqv/problem/xor/MakefileTransformerSmall.mk figs` will train the models used in the main experiments and it will plot the relative figures.

## Customizing Experiments
The code is organized to provide easily customizable experiments. To achieve this, we used the `click` python package to build a tree-like command line interface (CLI). Our CLI provides a command to choose dataset type, callbacks, loggers, loss function, model architecture, optimizer, and scheduler. For example, the command: 
```
python3 semeqv/cli.py  
		--seed 42  
	cli callbacks cdists --split "train" \
		--input_embedding_path  ./input \
		--output_embedding_path ./output \
	cli callbacks log --epoch_log_path ./valid.log --split "validation" \
	cli loss cross-entropy \
	cli xor dataset default --batch_size 100 --split "train" \
	cli xor dataset default --batch_size 100 --split "train" \
	cli xor model transformer \
		--layers 1 --heads 1 \
		--embedding_size 4 \
		--activation "gelu" \
	cli optimizers adamw --learning_rate 0.0001 \
	cli schedulers cosine noscheduler \
	cli savers default-saver --dirpath $(dir $@) \
	cli train
```
This command works by setting up the class `Trainer` in `Trainer.py` with a build pattern with methods like `set_loss_fn(self, value)` or `set_model(self, value)`. The final command `cli train` finally launches the training. 
Of course, there are many more options than the one shown here. The interested reader can launch the command `help` at any point of the previous command to inspect the possible command or options available to be specified in the place of `help`. We also encourage checking [click](https://click.palletsprojects.com/en/8.1.x/) for comprehensive documentation. 

## Docker
To ensure reproducibility in the future, we also provide a docker file to build an environment in which to run experiments. We suggest using docker with a GPU to run the experiments (see the [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)).  Once you have built the container with the command `make docker-build`, you can enter the docker environment with `make docker-run`. From there, we need can run the commands already discussed. Note that, the `docker-run` command automatically binds to the current working directory (make sure to run the command from the project root). Therefore, data will be generated outside the virtual environment.

## Bibtex
```
@inproceedings{bertolottitying,
  title={By Tying Embeddings You Are Assuming the Distributional Hypothesis},
  author={Bertolotti, Francesco and Cazzola, Walter},
  booktitle={Forty-first International Conference on Machine Learning}
}
```

