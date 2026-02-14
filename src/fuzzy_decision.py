# src/fuzzy_decision.py

from score_text import get_group_scores   # 🔗 integration point (Block 2 → Block 3)

# --------------------------------------------------
# 1. Linguistic thresholds (score → Low/Medium/High)
# --------------------------------------------------
def to_level(score):
    if score < 0.30:
        return "Low"
    elif score < 0.60:
        return "Medium"
    else:
        return "High"


# --------------------------------------------------
# 2. Level → intensity (used only for comparison)
# --------------------------------------------------
INTENSITY_VALUE = {
    "Low": 0.2,
    "Medium": 0.5,
    "High": 0.9
}


# --------------------------------------------------
# 3. Fixed priority (conflict resolution)
# --------------------------------------------------
PRIORITY = ["hate", "offensive", "mild_hate", "no_hate"]


# --------------------------------------------------
# 4. Aggregate fuzzy outputs
# --------------------------------------------------
def aggregate_levels(levels):
    """
    levels:
    {
        "hate": "Low",
        "offensive": "Medium",
        "mild_hate": "Medium",
        "no_hate": "Low"
    }
    """
    return {
        label: INTENSITY_VALUE[level]
        for label, level in levels.items()
    }


# --------------------------------------------------
# 5. Defuzzification (final decision)
# --------------------------------------------------
def defuzzify(aggregated):
    max_value = max(aggregated.values())

    candidates = [
        label for label, value in aggregated.items()
        if value == max_value
    ]

    for label in PRIORITY:
        if label in candidates:
            return label


# --------------------------------------------------
# 6. FULL FUZZY PIPELINE (Block 3.3)
# --------------------------------------------------
def fuzzy_decision(topic_scores):
    """
    topic_scores (from LDA / Block 2):
    {
        "hate": float,
        "offensive": float,
        "mild_hate": float,
        "no_hate": float
    }
    """

    # A. scores → linguistic levels
    levels = {k: to_level(v) for k, v in topic_scores.items()}

    # B. levels → fuzzy intensities
    aggregated = aggregate_levels(levels)

    # C. final crisp label
    final_label = defuzzify(aggregated)

    return {
        "levels": levels,
        "aggregated": aggregated,
        "final_label": final_label
    }


# --------------------------------------------------
# 7. REAL PIPELINE RUN (CONNECTED)
# --------------------------------------------------
if __name__ == "__main__":
    sentence = "Shut up, you fucking cunt!"

    # 🔹 Block 2 output (LDA)
    topic_scores = get_group_scores(sentence)

    # 🔹 Block 3 fuzzy logic
    result = fuzzy_decision(topic_scores)

    print("Sentence:", sentence)
    print("Topic scores:", topic_scores)
    print("Levels:", result["levels"])
    print("Aggregated:", result["aggregated"])
    print("Final label:", result["final_label"])
