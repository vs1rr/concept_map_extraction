"""
Global variables for the module
"""
import os

import spacy

nlp = spacy.load("en_core_web_lg")
API_KEY_GPT = ""
RND_SEED = 42

ROOT_DIR = os.getcwd()
SRC_DIR = os.path.join(ROOT_DIR + '/src')

# DATA DIR
DATA_DIR = os.path.join(ROOT_DIR + '/data')
REBEL_DIR = os.path.join(ROOT_DIR + '/rebel_fine_tuned/finetuned_rebel.pth')

# fine tuning rebel
REBEL = os.path.join(SRC_DIR + '/fine_tune_rebel')
BIO = os.path.join(REBEL + '/cm_biology.csv')
BIO_TEST = os.path.join(REBEL + '/cm_biology_test.csv')

# corpora Falke
CORPORA_FALKE = os.path.join(DATA_DIR + '/Corpora_Falke')
ACL = os.path.join(CORPORA_FALKE + '/ACL')
BIOLOGY = os.path.join(CORPORA_FALKE + '/Biology')
WIKI = os.path.join(CORPORA_FALKE + '/Wiki')
WIKI_TRAIN = os.path.join(WIKI + '/train/')
WIKI_TEST = os.path.join(WIKI + '/test')
WIKI_FINAL_TEST_DIR = os.path.join(WIKI_TEST + '/final_test')

# other corpora
CMAP_DIR = os.path.join(DATA_DIR + '/CMapSummaries')
CMAP_TEST_DIR = os.path.join(CMAP_DIR + '/test')
CMAP_TEST_GPT = os.path.join(CMAP_DIR + '/chat-gpt-test')
CMAP_FINAL_TEST_DIR = os.path.join(CMAP_TEST_DIR + '/final_test')
CMAP_TRAIN_DIR = os.path.join(CMAP_DIR + '/train')

list_dir = [DATA_DIR]
for x in list_dir:
    if not os.path.exists(x):
        os.makedirs(x)
