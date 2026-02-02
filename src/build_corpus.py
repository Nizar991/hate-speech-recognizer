from load_dataset import load_texts
from clean_text import clean_and_tokenize
from gensim.corpora import Dictionary, MmCorpus

import os
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "data", "data_huang_devansh.csv")

# where we save processed data
PROCESSED_DIR = os.path.join(BASE_DIR, "..", "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

if __name__ == "__main__":
    # 1. Load raw texts
    texts = load_texts(DATASET_PATH)

    # 2. Clean + tokenize each text
    tokenized_texts = [clean_and_tokenize(t) for t in texts]

    # 3. Build dictionary
    dictionary = Dictionary(tokenized_texts)

    # 4. Build Bag-of-Words corpus
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

    # 5. SAVE EVERYTHING (end of Block 1.4)
    dictionary.save(os.path.join(PROCESSED_DIR, "dictionary.gensim"))
    MmCorpus.serialize(os.path.join(PROCESSED_DIR, "corpus.mm"), corpus)

    with open(os.path.join(PROCESSED_DIR, "tokens.pkl"), "wb") as f:
        pickle.dump(tokenized_texts, f)

    # --- sanity check ---
    print("Number of documents:", len(corpus))
    print("Vocabulary size:", len(dictionary))

    # print("\nSample dictionary entries:")
    # print(dictionary[:10])

    # print("\nSample BoW document:")
    # print(corpus[0])
