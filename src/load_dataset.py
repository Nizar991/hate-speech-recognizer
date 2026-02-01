import pandas as pd

# Path to dataset (relative path)
DATASET_PATH = "../data/data_huang_devansh.csv"

def load_texts(csv_path):
    df = pd.read_csv(csv_path)

    # Ensure expected column exists
    if "Content" not in df.columns:
        raise ValueError("CSV must contain a 'Content' column")

    # Drop missing values and convert to list
    texts = (
    df["Content"]
    .dropna()
    .astype(str)
    .str.strip()
    .loc[lambda x: x != ""]
    .tolist()
)
    return texts


if __name__ == "__main__":
    texts = load_texts(DATASET_PATH)
    print(f"Loaded {len(texts)} texts")
    print("Sample text:")
    print(texts[0][:300])
