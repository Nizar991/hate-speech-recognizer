# src/fuzzy_scores.py

from score_text import get_group_scores


def normalize(scores):
    """
    Normalize group scores so they sum to 1.
    """
    total = sum(scores.values())
    return {k: v / total for k, v in scores.items()} if total else scores


def fuzzy_decision(scores):
    scores = normalize(scores)

    return {
        "hate_confidence": min(1.0, scores["hate"] + 0.3 * scores["offensive"]),
        "offensive_confidence": min(1.0, scores["offensive"] + 0.3 * scores["hate"]),
        "non_hate_confidence": scores["non_hate"]
    }


if __name__ == "__main__":
    sentence = "It’s interesting how some people can make the smallest things feel so important."

    # 🔗 Dynamic link to Block 2.4
    group_scores = get_group_scores(sentence)

    fuzzy = fuzzy_decision(group_scores)

    print("Sentence:", sentence)
    print("Group scores:", group_scores)
    print("Fuzzy scores:", fuzzy)
