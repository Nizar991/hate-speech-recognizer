from gensim.models import LdaModel
from gensim.corpora import Dictionary, MmCorpus
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "..", "data", "processed")

MODEL_PATH = os.path.join(PROCESSED_DIR, "lda_model.gensim")
DICT_PATH = os.path.join(PROCESSED_DIR, "dictionary.gensim")
CORPUS_PATH = os.path.join(PROCESSED_DIR, "corpus.mm")

if __name__ == "__main__":
    print("Loading LDA model, dictionary, and corpus...")
    lda_model = LdaModel.load(MODEL_PATH)
    dictionary = Dictionary.load(DICT_PATH)
    corpus = MmCorpus(CORPUS_PATH)

    # -----------------------------
    # PART 1: Topic → Word weights
    # -----------------------------
    print("\n=== TOPICS (word distributions) ===\n")
    topics = lda_model.print_topics(num_words=10)
    for topic_id, topic_words in topics:
        print(f"Topic {topic_id}:")
        print(topic_words)
        print("-" * 50)

    # --------------------------------
    # PART 2: Document → Topic weights
    # --------------------------------
    print("\n=== SAMPLE DOCUMENT TOPIC WEIGHTS ===\n")

    for doc_id, doc in enumerate(corpus[:5]):  # inspect first 5 docs
        print(f"Document {doc_id}:")
        doc_topics = lda_model.get_document_topics(doc, minimum_probability=0.0)
        for topic_id, prob in doc_topics:
            print(f"  Topic {topic_id}: {prob:.3f}")
        print("-" * 50)
