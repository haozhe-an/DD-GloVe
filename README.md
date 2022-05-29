# DD-GloVe
Repository for the implementation and evaluation of DD-GloVe, 
a train-time debiasing algorithm to learn GloVe word embeddings by leveraging **d**ictionary **d**efinitions.

## DD-GloVe Word Embeddings
Our trained embeddings are available [here](https://drive.google.com/drive/folders/1yqpBcqENLkPrzL1wfkw08GkO6VQ8m2tf?usp=sharing).

To load trained word embeddings, you may reference the code [here](https://github.com/haozhe-an/DD-GloVe/blob/aaf9850e20dff6705e26bbf6fd73e0b478f7f1ee/eval/python/evaluate.py#L138).

## Training
### Training corpus
Training corpus is not provided in this repository. We point out some public training corpus available online.

- wikipedia: https://huggingface.co/datasets/wikipedia
- text8: `wget https://data.deepai.org/text8.zip`

DD-GloVe and other baselines in this paper were trained on wikipedia corpus.

### Dictionary definitions
The released embeddings of DD-GloVe were trained using definitions from [Oxford dictionary](https://www.lexico.com/). We also conducted experiments using [WordNet](https://wordnet.princeton.edu/) and found that the dictionary content affect the embeddings qualities minimally.

### Running the code
`./demo.sh <use_def_loss> <lambda> <use_ortho_loss> <beta> <use_proj_loss> <gamma> <max_itr>`

where `use_def_loss`, `use_ortho_loss`, and `use_proj_loss` are either 1 or 0 to indicate using or not using each component of the loss, `lambda`, `beta`, and `gamma` are the weights for the loss term precedding them, and `max_itr` is the maximum number of iterations for training.

For example,
`./demo.sh 1 0.005 1 0.01 1 0.005 40` will train DD-GloVe with `use_def_loss=1, alpha(def_loss_weight)=0.01, use_ortho_loss=1, beta(ortho_loss_weight)=0.01, use_proj_loss=1, gamma(proj_loss_weight)=0.005, max_itr=40`.

This command will read the training corpus, produce vocab count, compute co-occurrences, get definitions of all vocab, train the word embeddings, and save them.

**IMPORTANT**

- You will need to modify line 79 and 81 in `./src/glove.c` to compute the correct bias directions given your own training corpus.
  - Line 79 needs the word indices of two initial seed words: `int SEED_WORD_1 = 19, SEED_WORD_2 = 43; // Modify them to your seed words indices in vocab.txt`.
  - Line 81 defines the greatest word index up to which the words will be considered as candidates words to compute the bias direction: `long long cap = 30000; // This number must be smaller than vocab size`.
- You will need to modify line 23 in `./demo.sh` to read your training corpus.

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
## Citation
Haozhe An, Xiaojiang Liu, and Jian Zhang. "Learning Bias-reduced Word Embeddings Using Dictionary Definitions." Findings of ACL. 2022.
```
@inproceedings{an-etal-2022-learning,
    title = "Learning Bias-reduced Word Embeddings Using Dictionary Definitions",
    author = "An, Haozhe  and
      Liu, Xiaojiang  and
      Zhang, Donald",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2022",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-acl.90",
    pages = "1139--1152",
    abstract = "Pre-trained word embeddings, such as GloVe, have shown undesirable gender, racial, and religious biases. To address this problem, we propose DD-GloVe, a train-time debiasing algorithm to learn word embeddings by leveraging $\underline{d}$ictionary $\underline{d}$efinitions. We introduce dictionary-guided loss functions that encourage word embeddings to be similar to their relatively neutral dictionary definition representations. Existing debiasing algorithms typically need a pre-compiled list of seed words to represent the bias direction, along which biased information gets removed. Producing this list involves subjective decisions and it might be difficult to obtain for some types of biases. We automate the process of finding seed words: our algorithm starts from a single pair of initial seed words and automatically finds more words whose definitions display similar attributes traits. We demonstrate the effectiveness of our approach with benchmark evaluations and empirical analyses. Our code is available at https://github.com/haozhe-an/DD-GloVe.",
}
```

