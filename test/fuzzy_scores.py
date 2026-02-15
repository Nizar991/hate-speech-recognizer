# src/fuzzy_scores.py

from score_text import get_group_scores

def normalize(scores):
    """
    Normalize group scores so they sum to 1.
    """
    total = sum(scores.values())
    return {k: v / total for k, v in scores.items()} if total else scores

def fuzzy_decision(scores):
    """
    Prepare fuzzy confidence signals for Part 3.
    No cross-influence. Clean separation.
    """
    scores = normalize(scores)

    hate = scores["hate"]
    offensive = scores["offensive"]
    non_hate = scores["non_hate"]

    # Mild hate = weak negativity that is not strong hate
    mild_hate = max(0.0, offensive - hate)

    return {
        "hate_confidence": hate,
        "offensive_confidence": offensive,
        "mild_hate_confidence": mild_hate,
        "no_hate_confidence": non_hate
    }

if __name__ == "__main__":
    sentence = "It’s interesting how some people can make the smallest things feel so important."

    # 🔗 Dynamic link to Block 2.4
    group_scores = get_group_scores(sentence)

    fuzzy = fuzzy_decision(group_scores)

    print("Sentence:", sentence)
    print("Group scores:", group_scores)
    print("Fuzzy scores:", fuzzy)
