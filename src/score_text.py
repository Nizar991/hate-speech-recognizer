from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "..", "data", "processed")

MODEL_PATH = os.path.join(PROCESSED_DIR, "lda_model.gensim")
DICT_PATH = os.path.join(PROCESSED_DIR, "dictionary.gensim")

# Topic groups (from Block 2.3)
HATE_TOPICS = {1, 6}
POTENTIAL_HATE_TOPICS = {4, 9}
OFFENSIVE_TOPICS = {5, 7}

# Strengthen hate signal (numeric only)
HATE_WEIGHT = 1.5

def clean(text):
    return [
        w for w in simple_preprocess(text, deacc=True)
        if w not in STOPWORDS
    ]

if __name__ == "__main__":
    lda = LdaModel.load(MODEL_PATH)
    dictionary = Dictionary.load(DICT_PATH)

    sentence = "Shut up, you fucking cunt!"
    tokens = clean(sentence)
    bow = dictionary.doc2bow(tokens)

    topic_probs = dict(lda.get_document_topics(bow))

    scores = {
        "hate": HATE_WEIGHT * sum(topic_probs.get(t, 0) for t in HATE_TOPICS),
        "potential_hate": sum(topic_probs.get(t, 0) for t in POTENTIAL_HATE_TOPICS),
        "offensive": sum(topic_probs.get(t, 0) for t in OFFENSIVE_TOPICS),
        "non_hate": sum(
            p for t, p in topic_probs.items()
            if t not in HATE_TOPICS | POTENTIAL_HATE_TOPICS | OFFENSIVE_TOPICS
        )
    }

    print("Cleaned tokens:", tokens)
    print("Topic probs:", topic_probs)
    print("Group scores:", scores)
