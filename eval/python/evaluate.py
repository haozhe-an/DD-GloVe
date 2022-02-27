import argparse
import numpy as np
from utils import p_value_exhust, effect_size
from utils import WEAT_words, limit_vocab
import codecs
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_file', default='vocab.txt', type=str)
    parser.add_argument('--vectors_file', default='vectors.txt', type=str)
    args = parser.parse_args()

    with open(args.vocab_file, 'r') as f:
        words = [x.rstrip().split(' ')[0] for x in f.readlines()]
    with open(args.vectors_file, 'r') as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
            if len(vals) < 3:
                continue
            vectors[vals[0]] = [float(x) for x in vals[1:]]

    vocab_size = len(words)
    vocab = {w: idx for idx, w in enumerate(words)}
    ivocab = {idx: w for idx, w in enumerate(words)}

    vector_dim = len(vectors[ivocab[0]])
    W = np.zeros((vocab_size, vector_dim))
    for word, v in vectors.items():
        if word == '<unk>':
            continue
        W[vocab[word], :] = v

    # normalize each word vector to unit length
    W_norm = np.zeros(W.shape)
    d = (np.sum(W ** 2, 1) ** (0.5))
    W_norm = (W.T / d).T
    evaluate_vectors(W_norm, vocab)

    gender_specific = []
    with open('male_word_file.txt') as f:
        for l in f:
            gender_specific.append(l.strip())
    with open('female_word_file.txt') as f:
        for l in f:
            gender_specific.append(l.strip())

    with codecs.open('gender_specific_full.json') as f:
        gender_specific.extend(json.load(f))

    wv_glove_full, w2i_glove_full, vocab_glove_full = load_glove(args.vectors_file)
    vocab_glove, wv_glove, w2i_glove = limit_vocab(wv_glove_full, w2i_glove_full, vocab_glove_full,
                                                   exclude=gender_specific)
    WEAT_test(wv_glove, w2i_glove, vocab_glove)


def evaluate_vectors(W, vocab):
    """Evaluate the trained word vectors on a variety of tasks"""

    filenames = [
        'capital-common-countries.txt', 'capital-world.txt', 'currency.txt',
        'city-in-state.txt', 'family.txt', 'gram1-adjective-to-adverb.txt',
        'gram2-opposite.txt', 'gram3-comparative.txt', 'gram4-superlative.txt',
        'gram5-present-participle.txt', 'gram6-nationality-adjective.txt',
        'gram7-past-tense.txt', 'gram8-plural.txt', 'gram9-plural-verbs.txt',
        ]
    prefix = './embeddings_eval/data/question-data/'

    # to avoid memory overflow, could be increased/decreased
    # depending on system and vocab size
    split_size = 100

    correct_sem = 0; # count correct semantic questions
    correct_syn = 0; # count correct syntactic questions
    correct_tot = 0 # count correct questions
    count_sem = 0; # count all semantic questions
    count_syn = 0; # count all syntactic questions
    count_tot = 0 # count all questions
    full_count = 0 # count all questions, including those with unknown words

    for i in range(len(filenames)):
        with open('%s/%s' % (prefix, filenames[i]), 'r') as f:
            full_data = [line.rstrip().split(' ') for line in f]
            full_count += len(full_data)
            data = [x for x in full_data if all(word in vocab for word in x)]

        if len(data) == 0:
            print("ERROR: no lines of vocab kept for %s !" % filenames[i])
            print("Example missing line:", full_data[0])
            continue

        indices = np.array([[vocab[word] for word in row] for row in data])
        ind1, ind2, ind3, ind4 = indices.T

        predictions = np.zeros((len(indices),))
        num_iter = int(np.ceil(len(indices) / float(split_size)))
        for j in range(num_iter):
            subset = np.arange(j*split_size, min((j + 1)*split_size, len(ind1)))

            pred_vec = (W[ind2[subset], :] - W[ind1[subset], :]
                +  W[ind3[subset], :])
            #cosine similarity if input W has been normalized
            dist = np.dot(W, pred_vec.T)

            for k in range(len(subset)):
                dist[ind1[subset[k]], k] = -np.Inf
                dist[ind2[subset[k]], k] = -np.Inf
                dist[ind3[subset[k]], k] = -np.Inf

            # predicted word index
            predictions[subset] = np.argmax(dist, 0).flatten()

        val = (ind4 == predictions) # correct predictions
        count_tot = count_tot + len(ind1)
        correct_tot = correct_tot + sum(val)
        if i < 5:
            count_sem = count_sem + len(ind1)
            correct_sem = correct_sem + sum(val)
        else:
            count_syn = count_syn + len(ind1)
            correct_syn = correct_syn + sum(val)

        print("%s:" % filenames[i])
        print('ACCURACY TOP1: %.2f%% (%d/%d)' %
            (np.mean(val) * 100, np.sum(val), len(val)))

    print('Questions seen/total: %.2f%% (%d/%d)' %
        (100 * count_tot / float(full_count), count_tot, full_count))
    print('Semantic accuracy: %.2f%%  (%i/%i)' %
        (100 * correct_sem / float(count_sem), correct_sem, count_sem))
    print('Syntactic accuracy: %.2f%%  (%i/%i)' %
        (100 * correct_syn / float(count_syn), correct_syn, count_syn))
    print('Total accuracy: %.2f%%  (%i/%i)' % (100 * correct_tot / float(count_tot), correct_tot, count_tot))


def load_glove(path):
    if 'bin' in path:
        import gensim.models
        model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
        words = sorted([w for w in model.vocab], key=lambda w: model.vocab[w].index)
        vecs = [model[w] for w in words]

        vecs = np.array(vecs, dtype='float32')

        wv = vecs
        vocab = words
        w2i = {word: index for index, word in enumerate(words)}
        print(len(vocab), wv.shape, len(w2i))
    else:
        with open(path) as f:
            lines = f.readlines()
            wv = []
            vocab = []
            for line in lines:
                tokens = line.strip().split(' ')
                if not len(tokens) == 301:
                    continue
                vocab.append(tokens[0])
                wv.append([float(elem) for elem in tokens[1:]])

            w2i = {w: i for i, w in enumerate(vocab)}
            for i in range(len(wv)):
                for ele in wv[i]:
                    if not isinstance(ele, float):
                        print(wv[i])
            wv = np.array(wv).astype(float)
            print(len(vocab), wv.shape, len(w2i))
    return wv, w2i, vocab


def WEAT_test(wv, w2i, vocab):
    A = [item.lower() for item in WEAT_words['A']]
    B = [item.lower() for item in WEAT_words['B']]
    C1 = [item.lower() for item in WEAT_words['C']]
    D1 = [item.lower() for item in WEAT_words['D']]
    C2 = [item.lower() for item in WEAT_words['E']]
    D2 = [item.lower() for item in WEAT_words['F']]
    C3 = [item.lower() for item in WEAT_words['G']]
    D3 = [item.lower() for item in WEAT_words['H']]

    for C, D in zip([C1, C2, C3], [D1, D2, D3]):
        print('WEAT test words:')
        print(A)
        print(B)
        print(C)
        print(D)
        print('effect size: {}'.format(effect_size(A, B, C, D, wv, w2i, vocab)))
        print('p value: {}'.format(p_value_exhust(A, B, C, D, wv, w2i, vocab)))
        print()


if __name__ == "__main__":
    main()
