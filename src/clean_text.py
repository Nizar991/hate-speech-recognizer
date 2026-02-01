import re
import spacy

# Load full English model (for lemmatization + stopwords)
nlp = spacy.load("en_core_web_sm")

def clean_and_tokenize(text: str):
    """
    Clean text and return list of lemmatized tokens
    """
    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Process with spaCy
    doc = nlp(text)

    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and token.is_alpha
    ]

    return tokens


if __name__ == "__main__":
    sample = "I don't like running cars and I’ll never stop!"
    print("Tokens:", clean_and_tokenize(sample))
