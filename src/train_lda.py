from gensim.models import LdaModel
from gensim.corpora import Dictionary, MmCorpus
import os

# ---- paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "..", "data", "processed")

DICT_PATH = os.path.join(PROCESSED_DIR, "dictionary.gensim")
CORPUS_PATH = os.path.join(PROCESSED_DIR, "corpus.mm")
MODEL_PATH = os.path.join(PROCESSED_DIR, "lda_model.gensim")

# ---- Block 2.1 config ----
NUM_TOPICS = 10
PASSES = 5
RANDOM_STATE = 42

if __name__ == "__main__":
    print("Loading dictionary and corpus...")
    dictionary = Dictionary.load(DICT_PATH)
    corpus = MmCorpus(CORPUS_PATH)

    print("Training LDA model...")
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=NUM_TOPICS,
        passes=PASSES,
        random_state=RANDOM_STATE
    )

    print("Saving LDA model...")
    lda_model.save(MODEL_PATH)

    print("LDA training complete ✅")
