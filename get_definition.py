import nltk
from nltk import word_tokenize
import numpy as np
import argparse
import requests
from tqdm import tqdm
import string

nltk.download('punkt')
nltk.download('wordnet')

parser = argparse.ArgumentParser()
parser.add_argument('--vocab_file', default='vocab.txt', type=str)
parser.add_argument('--max_vocab_size', default=400000, type=int)
args = parser.parse_args()

train_words = []
word_to_id = {}

print('Reading from vocab file...')
with open(args.vocab_file, 'r') as f:
    for idx, line in enumerate(f.readlines()):
        train_words.append((line.strip().split()[0]))
        if len(train_words) == args.max_vocab_size: break
word_to_id = {word: i for i, word in enumerate(train_words)}

print('check length of word lists: {}'.format(len(train_words)))

stopwords = set()
with open('stopwords.txt') as f:
    for line in f.readlines():
        stopwords.add(line.strip())

print('Done reading stopwords file.')

def_history = {}
for idx, word in enumerate(train_words):
    def_history[idx] = []

for idx, word in tqdm(enumerate(train_words)):
    if idx > 800:
        break

    url_prefix = 'https://api.dictionaryapi.dev/api/v2/entries/en_US/'

    r = requests.get(url=url_prefix + word)
    try:
        data = r.json()
        for datum in data:
            meanings = datum['meanings']
            for m in meanings:
                for d in m['definitions']:
                    definition = d['definition']
                    for def_w in word_tokenize(definition):
                        def_w = def_w.lower()
                        if def_w != word and def_w in train_words and not def_w in stopwords and not def_w in string.punctuation:
                            def_history[idx].append(word_to_id[def_w])         
    except:
        print("An exception occured while getting the definition for {}.".format(word))
        continue

print('check length of def_history', len(def_history))
to_write = []
file_obj = open('definitions.dat', 'wb')
to_write.append(len(def_history))
for w_id, def_ids in def_history.items():
    to_write.append(w_id)
    to_write.append(len(def_ids))
    to_write.extend(def_ids)
np.array(to_write, dtype=np.int32).tofile(file_obj)
file_obj.close()

with open('definitions.txt', 'w') as f:
    for word_id, definitions in def_history.items():
        content = '{}\t{}\n'.format(word_id, definitions)
        f.write(content)
