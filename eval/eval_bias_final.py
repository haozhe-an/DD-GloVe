import codecs
import json
import math
import operator
import os

import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import wordnet
from scipy import spatial
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

from utils import WEAT_words, p_value_exhust, effect_size
from utils import extract_vectors, limit_vocab, doPCA, drop

from eval import evaluate_cate, evaluate_ana

BASE_FILE = 'vectors_glove.txt'

DEBIAS_FILES = ['vectors_gn.txt',
                'vectors_dict_debias.txt',
                'vectors_ddglove_gender.txt',
                'vectors_ddglove_race.txt']

only_in_wn = True
plot_content = 'y'
random_state = 0
tsne_random_state = 0
size = 500
MALE = 'he'
FEMALE = 'she'

if not os.path.exists("./figures"):
	os.makedirs("./figures")

with open('figures/setting.txt', 'w') as f:
    f.write('only_in_wn: {}\n'.format(only_in_wn))
    f.write('random_state: {}\ttsne_random_state:{}\n'.format(random_state, tsne_random_state))
    f.write('size: {}\tmale word: {}\tfemale word: {}\n'.format(size, MALE, FEMALE))
    f.write(('plot_content: {}'.format(plot_content)))


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
            for line in tqdm(lines):
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


def simi(a, b):
    ret = 1 - spatial.distance.cosine(a, b)
    if math.isnan(ret):
        ret = 0
    return ret


def visualize_tsne(words, vectors, y_true, y_pred, ax, title, random_state):
    # perform TSNE
    X_embedded = TSNE(n_components=2, random_state=random_state).fit_transform(vectors)

    for x, p, y in zip(X_embedded, y_pred, y_true):
        if plot_content == 'y':
            if y:
                ax.scatter(x[0], x[1], marker='.', c='c')
            else:
                ax.scatter(x[0], x[1], marker='x', c='darkviolet')
        elif plot_content == 'p':
            if p:
                ax.scatter(x[0], x[1], marker='.', c='c')
            else:
                ax.scatter(x[0], x[1], marker='x', c='darkviolet')
    return ax


def cluster_and_visualize(words, X1, title, random_state, tsne_random_state, y_true, num=2,
                          file_name=''):
    kmeans_1 = KMeans(n_clusters=num, random_state=random_state).fit(X1)
    y_pred_1 = kmeans_1.predict(X1)

    print('Testing reclassifying...')
    print('Current file is {}'.format(file_name))

    print('labelled 1 by kmeans', sum(y_pred_1))

    correct = [1 if item1 == item2 else 0 for (item1, item2) in zip(y_true, y_pred_1)]
    acc = max(sum(correct) / float(len(correct)), 1 - sum(correct) / float(len(correct)))
    print('accuracy', acc)
    print()

    fig, axs = plt.subplots(1, 1, figsize=(4, 4))
    visualize_tsne(words, X1, y_true, y_pred_1, axs, title, tsne_random_state)
    fig.suptitle('Accuracy: {}'.format(acc), fontsize=14)
    fig.tight_layout()
    fig.savefig('figures/fig_reclassify_{}.pdf'.format(file_name.strip('.txt').replace('/', '_')))
    plt.close(fig)

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

    positive = 'freedom, health, love, peace, cheer, friend, heaven, loyal, pleasure, diamond, gentle, honest, lucky, rainbow, diploma, gift, honor, miracle, sunrise, family, happy, laughter, paradise, vacation'.split(
        ', ')[:10]
    negative = 'abuse, crash, filth, sickness, accident, death, grief, poison, stink, assault, disaster, hatred, pollute, tragedy, bomb, divorce, jail, poverty, ugly, evil, kill, rotten, vomit, agony'.split(
        ', ')[:10]

    eu_american_names = 'adam, chip, harry, josh, roger, alan, frank, ian, justin, ryan'.split(', ')
    af_american_names = 'alonzo, jamel, theo, alphonse, jerome, leroy, torrance, darnell, lamar, lionel'.split(', ')
    print('WEAT test words:')
    print(eu_american_names)
    print(af_american_names)
    print(positive)
    print(negative)
    print('effect size: {}'.format(effect_size(eu_american_names, af_american_names, positive, negative, wv, w2i, vocab)))
    print('p value: {}'.format(p_value_exhust(eu_american_names, af_american_names, positive, negative, wv, w2i, vocab)))
    print()

    old_name = 'ethel, bernice, gertrude, agnes, cecil, wilbert, mortimer, edgar'.split(', ')
    young_name = 'tiffany, michelle, cindy, kristy, brad, eric, joey, billy'.split(', ')
    positive = positive[:8]
    negative = negative[:8]
    print('WEAT test words:')
    print(old_name)
    print(young_name)
    print(positive)
    print(negative)
    print('effect size: {}'.format(effect_size(eu_american_names, af_american_names, positive, negative, wv, w2i, vocab)))
    print('p value: {}'.format(p_value_exhust(eu_american_names, af_american_names, positive, negative, wv, w2i, vocab)))
    print()

    flowers = 'aster, clover, hyacinth, marigold, poppy, azalea, crocus, iris, orchid, rose, bluebell, daffodil, lilac, pansy, tulip, buttercup, daisy, lily, peony, violet, carnation, magnolia, petunia, zinnia'.split(', ')[:10]
    insects = 'ant, caterpillar, flea, locust, spider, bedbug, centipede, fly, maggot, tarantula, bee, cockroach, gnat, mosquito, termite, beetle, cricket, hornet, moth, wasp, blackfly, dragonfly, horsefly, roach'.split(', ')[:10]
    print('WEAT test words:')
    print(flowers)
    print(insects)
    print(positive)
    print(negative)
    print('effect size: {}'.format(effect_size(flowers, insects, positive, negative, wv, w2i, vocab)))
    print('p value: {}'.format(p_value_exhust(flowers, insects, positive, negative, wv, w2i, vocab)))
    print()


def my_pca(wv):
    wv_mean = np.mean(np.array(wv), axis=0)
    wv_hat = np.zeros(wv.shape).astype(float)

    for i in range(len(wv)):
        wv_hat[i, :] = wv[i, :] - wv_mean

    main_pca = PCA()
    main_pca.fit(wv_hat)

    return main_pca


def hard_debias(wv, w2i, w2i_partial, vocab_partial, component_ids, main_pca):
    D = []

    for i in component_ids:
        D.append(main_pca.components_[i])

    # get rid of frequency features
    wv_f = np.zeros((len(vocab_partial), wv.shape[1])).astype(float)

    for i, w in enumerate(vocab_partial):
        u = wv[w2i[w], :]
        sub = np.zeros(u.shape).astype(float)
        for d in D:
            sub += np.dot(np.dot(np.transpose(d), u), d)
        wv_f[w2i_partial[w], :] = wv[w2i[w], :] - sub - wv_mean

    gender_directions = list()
    for gender_word_list in [definitional_pairs]:
        gender_directions.append(doPCA(gender_word_list, wv_f, w2i_partial).components_[0])

    wv_debiased = np.zeros((len(vocab_partial), len(wv_f[0, :]))).astype(float)
    for i, w in enumerate(vocab_partial):
        u = wv_f[w2i_partial[w], :]
        for gender_direction in gender_directions:
            u = drop(u, gender_direction)
            wv_debiased[w2i_partial[w], :] = u

    return wv_debiased


def cluster_and_visualize_dhd(words, X, random_state, y_true, num=2):
    kmeans = KMeans(n_clusters=num, random_state=random_state).fit(X)
    y_pred = kmeans.predict(X)
    correct = [1 if item1 == item2 else 0 for (item1, item2) in zip(y_true, y_pred)]
    preci = max(sum(correct) / float(len(correct)), 1 - sum(correct) / float(len(correct)))
    print('precision', preci)

    return kmeans, y_pred, X, preci


gender_specific = []
with open('male_word_file.txt') as f:
    for l in f:
        gender_specific.append(l.strip())
with open('female_word_file.txt') as f:
    for l in f:
        gender_specific.append(l.strip())

with codecs.open('gender_specific_full.json') as f:
    gender_specific.extend(json.load(f))

exclude_words = gender_specific

definitional_pairs = [['she', 'he'], ['herself', 'himself'], ['her', 'his'], ['daughter', 'son'],
                      ['girl', 'boy'], ['mother', 'father'], ['woman', 'man'], ['mary', 'john'],
                      ['gal', 'guy'], ['female', 'male']]
definitional_words = []
for pair in definitional_pairs:
    for word in pair:
        definitional_words.append(word)

print('====== begin testing base file ======\n')
wv_glove_full, w2i_glove_full, vocab_glove_full = load_glove(BASE_FILE)

evaluate_ana(wv_glove_full, w2i_glove_full, vocab_glove_full)
evaluate_cate(wv_glove_full, w2i_glove_full, vocab_glove_full)

vocab_glove, wv_glove, w2i_glove = limit_vocab(wv_glove_full, w2i_glove_full, vocab_glove_full, exclude=gender_specific)

he_embed_glove = wv_glove_full[w2i_glove_full[MALE]]
she_embed_glove = wv_glove_full[w2i_glove_full[FEMALE]]
glove_bias = {}
for word in w2i_glove.keys():
    u = wv_glove[w2i_glove[word]]
    glove_bias[word] = simi(u, he_embed_glove) - simi(u, she_embed_glove)
glove_sorted_bias = sorted(glove_bias.items(), key=operator.itemgetter(1))

female, male = [], []

if only_in_wn:
    for idx in range(len(glove_sorted_bias)):
        if wordnet.synsets(glove_sorted_bias[idx][0]):
            female.append(glove_sorted_bias[idx][0])
        if len(female) == size: break

    for idx in range(len(glove_sorted_bias)):
        if wordnet.synsets(glove_sorted_bias[-idx][0]):
            male.append(glove_sorted_bias[-idx][0])
        if len(male) == size: break
else:
    female = [item[0] for item in glove_sorted_bias[:size]]
    male = [item[0] for item in glove_sorted_bias[-size:]]

y_true = [1] * size + [0] * size

with open('figures/top_biased_male.txt', 'w') as f:
    f.write(str(male))

with open('figures/top_biased_female.txt', 'w') as f:
    f.write(str(female))


cluster_and_visualize(male + female, extract_vectors(male + female, wv_glove, w2i_glove),
                      'GloVe', random_state, tsne_random_state, y_true, file_name=BASE_FILE)

WEAT_test(wv_glove, w2i_glove, vocab_glove)

print('====== done with base file ======\n')


print('====== begin testing double hard debias ======\n')
main_pca = my_pca(wv_glove_full)
wv_mean = np.mean(np.array(wv_glove_full), axis=0)

c_vocab = list(set(male + female + [word for word in definitional_words if word in w2i_glove_full]))
c_w2i = dict()
for idx, w in enumerate(c_vocab):
    c_w2i[w] = idx

precisions = []

best_component = None
best_prec = 1
for component_id in range(20):

    print('component id: ', component_id)

    wv_debiased = hard_debias(wv_glove_full, w2i_glove_full, w2i_partial=c_w2i, vocab_partial=c_vocab,
                              component_ids=[component_id], main_pca=main_pca)
    _, _, _, preci = cluster_and_visualize_dhd(male + female,
                                               extract_vectors(male + female, wv_debiased, c_w2i), 1, y_true)
    precisions.append(preci)

    if preci < best_prec:
        best_prec = preci
        best_component = component_id

print('Component to drop is {}'.format(best_component))
wv_debiased = hard_debias(wv_glove_full, w2i_glove_full, w2i_partial=w2i_glove_full, vocab_partial=vocab_glove_full,
                          component_ids=[best_component], main_pca=main_pca)

# np.save('./dhd_baseline', wv_debiased)
# with open('vectors_my_dhd.txt', 'w') as f:
#     f.write('{} 300\n'.format(len(vocab_glove_full)))
#     for word, vector in zip(vocab_glove_full, wv_glove_full):
#         f.write('{} '.format(word))
#         for v in vector:
#             f.write(('{} '.format(v)))
#         f.write('\n')

evaluate_ana(wv_debiased, w2i_glove_full, vocab_glove_full)
evaluate_cate(wv_debiased, w2i_glove_full, vocab_glove_full)

vocab_glove_dhd, wv_glove_dhd, w2i_glove_dhd = limit_vocab(wv_debiased, w2i_glove_full, vocab_glove_full,
                                                           exclude=gender_specific)
cluster_and_visualize(male + female, extract_vectors(male + female, wv_glove_dhd, w2i_glove_dhd),
                      'GloVe', random_state, tsne_random_state, y_true, file_name='DHD+{}'.format(BASE_FILE))
WEAT_test(wv_glove_dhd, w2i_glove_dhd, vocab_glove_dhd)

print('====== done with double hard debias ======\n')

for debias_file in DEBIAS_FILES:
    print('====== begin testing {} ======\n'.format(debias_file))
    wv_def_l1_full, w2i_def_l1_full, vocab_def_l1_full = load_glove(debias_file)

    evaluate_ana(wv_def_l1_full, w2i_def_l1_full, vocab_def_l1_full)
    evaluate_cate(wv_def_l1_full, w2i_def_l1_full, vocab_def_l1_full)

    vocab_def_l1, wv_def_l1, w2i_def_l1 = limit_vocab(wv_def_l1_full, w2i_def_l1_full, vocab_def_l1_full,
                                                      exclude=gender_specific)

    cluster_and_visualize(male + female, extract_vectors(male + female, wv_def_l1, w2i_def_l1),
                          'GloVe', random_state, tsne_random_state, y_true, file_name=debias_file)


    WEAT_test(wv_def_l1, w2i_def_l1, vocab_def_l1)

    print('====== done with {} ======\n'.format(debias_file))
