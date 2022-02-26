# DD-GloVe
Repository for the implementation and evaluation of DD-GloVe, 
a train-time debiasing algorithm to learn GloVe word embeddings by leveraging **d**ictionary **d**efinitions.

Our work is to appear in Findings of ACL 2022.

## DD-GloVe Word Embeddings
Our trained embeddings are available [here](https://drive.google.com/drive/folders/1yqpBcqENLkPrzL1wfkw08GkO6VQ8m2tf?usp=sharing).

## Evaluation
The folder `embeddings_eval` contains our code to evaluate the qualities of DD-GloVe and GloVe embeddings trained using other debiasing algorithms.
Please download our trained DD-GloVe embeddings and other baselines [here](https://drive.google.com/drive/folders/1yqpBcqENLkPrzL1wfkw08GkO6VQ8m2tf?usp=sharing).
Note that the evaluation code performs double hard debias on the fly. Place the word embedding files into the folder `eval` (i.e. the same level as `eval_bias_final.py`.

To reporduce the evaluation results in our paper, please follow these guidelines. 
### Environments
We used the following packages (and their versions) while running these evluations.

| Package | Version|
| ----------- | ----------- |
| python  |                  3.7.9 |
| numpy   |                  1.19.2 |
| scipy   |                  1.6.1 |
| scikit-learn  |            0.23.2 |
|Word Embedding Benchmarks | [Install here](https://github.com/kudkudak/word-embeddings-benchmarks) |

### Running the code
`python eval_bias_final.py`

WEAT and semantic meaning preservations results will be printed out to the console. Neighborhood metric plots will be saved in the folder `figures`.

### Coreference resolution
We followed this [demo page of AllenNLP](https://demo.allennlp.org/coreference-resolution) to run evaluations on coreference resolution.
The dataset [OntoNotes Release 5.0](https://catalog.ldc.upenn.edu/LDC2013T19) is needed for training. 
Bias evaluation is conducted by using [WinoBias](https://github.com/uclanlp/corefBias/tree/master/WinoBias/wino).

To set up the training environment for the coreference models, we ran the commands below.
```
pip install allennlp==2.1.0 allennlp-models==2.1.0 
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

The specifications of baseline model by Lee et al. (2017) are avaliable [here](https://raw.githubusercontent.com/allenai/allennlp-models/main/training_config/coref/coref.jsonnet). Download the `jsonnet` file and run these commands to train and evaluate.
```
allennlp train <model_jsonnet> -s <path_to_output>
allennlp evaluate <path_to_model> <path_to_dataset> --cuda-device 0 --output-file <path_to_output_file>
```
For example,
```
allennlp train coref.jsonnet -s coref_model
allennlp evaluate coref_model/model.tar.gz test.english.v4_gold_conll --cuda-device 0 --output-file eval_output_base
```
