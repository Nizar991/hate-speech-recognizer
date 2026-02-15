from load_dataset import load_texts
from clean_text import clean_and_tokenize

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "data", "data_huang_devansh.csv")

if __name__ == "__main__":
    texts = load_texts(DATASET_PATH)

    cleaned_texts = [clean_and_tokenize(t) for t in texts]

    print("Original sample:")
    print(texts[0][:300])
    print("\nCleaned sample:")
    print(cleaned_texts[0][:300])
