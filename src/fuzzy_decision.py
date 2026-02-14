# src/fuzzy_decision.py

from score_text import get_group_scores   # 🔗 Block 2 → Block 3 integration

# --------------------------------------------------
# 1. Linguistic thresholds (score → Low/Medium/High)
# --------------------------------------------------
def to_level(score: float) -> str:
    if score < 0.30:
        return "Low"
    elif score < 0.60:
        return "Medium"
    else:
        return "High"

# --------------------------------------------------
# 2. Level → intensity (comparison only)
# --------------------------------------------------
INTENSITY_VALUE = {
    "Low": 0.2,
    "Medium": 0.5,
    "High": 0.9
}

# --------------------------------------------------
# 3. Global dominance order (tie-breaking)
# --------------------------------------------------
PRIORITY = ["hate", "offensive", "mild_hate", "no_hate"]

# --------------------------------------------------
# 4. Aggregate linguistic levels
# --------------------------------------------------
def aggregate_levels(levels: dict) -> dict:
    return {
        label: INTENSITY_VALUE[level]
        for label, level in levels.items()
    }

# --------------------------------------------------
# 5. Defuzzification (final decision logic)
# --------------------------------------------------
def defuzzify(levels: dict, aggregated: dict):
    # 🔥 Fuse condition → system failure
    if all(level == "Low" for level in levels.values()):
        return None

    # ✅ STRICT no_hate VALIDATION RULE
    if (
        levels["no_hate"] in ("Medium", "High")
        and levels["hate"] == "Low"
        and levels["offensive"] == "Low"
        and levels["mild_hate"] == "Low"
    ):
        return "no_hate"

    # ❌ Otherwise, no_hate is suppressed
    filtered = {
        k: v for k, v in aggregated.items()
        if k != "no_hate"
    }

    max_value = max(filtered.values())

    candidates = [
        label for label, value in filtered.items()
        if value == max_value
    ]

    # Apply global dominance
    for label in PRIORITY:
        if label in candidates:
            return label

    return None  # safety fallback (should never hit)

# --------------------------------------------------
# 6. FULL FUZZY PIPELINE (Block 3.3)
# --------------------------------------------------
def fuzzy_decision(topic_scores: dict) -> dict:
    """
    topic_scores (from LDA / Block 2):
    {
        "hate": float,
        "offensive": float,
        "non_hate": float
    }
    """

    # --- Step A: base linguistic levels ---
    hate_level = to_level(topic_scores["hate"])
    offensive_level = to_level(topic_scores["offensive"])
    no_hate_level = to_level(topic_scores["non_hate"])

    # --- Step B: rule-based mild hate ---
    # Mild hate = light negativity without strong hostility
    if hate_level == "Low" and offensive_level in ("Low", "Medium"):
        mild_hate_level = "Medium"
    else:
        mild_hate_level = "Low"

    levels = {
        "hate": hate_level,
        "offensive": offensive_level,
        "mild_hate": mild_hate_level,
        "no_hate": no_hate_level
    }

    # --- Step C: aggregation ---
    aggregated = aggregate_levels(levels)

    # --- Step D: defuzzification ---
    final_label = defuzzify(levels, aggregated)

    return {
        "levels": levels,
        "aggregated": aggregated,
        "final_label": final_label
    }

# --------------------------------------------------
# 7. REAL PIPELINE RUN (CONNECTED)
# --------------------------------------------------
if __name__ == "__main__":

    sentences = [
        "All those filthy immigrants need to be gassed like the vermin they are, fucking up our pure nation.",
        "Bitches ain't shit but hoes and tricks, deserve to get slapped around till they learn their place.",
        "Faggots are diseased freaks that should be hunted down and burned at the stake.",
        "Niggers are subhuman apes, lynch 'em all and send 'em back to the jungle.",
        "Ragheads are bomb-wearing savages, nuke their shithole countries and wipe out their kind.",
        "That dumbass politician is a total retard, couldn't find his ass with both hands.",
        "Lard-ass fatties are gross pigs who stuff their faces and whine about health.",
        "Senile old geezers are worthless drains, just drop dead and stop wasting oxygen.",
        "Your cult religion is for braindead sheep, wake the fuck up or stay deluded.",
        "Bitch drivers suck balls, always causing wrecks 'cause they can't handle the wheel.",
        "I usually avoid hiring folks from those areas, they just don't vibe with the team culture.",
        "That part of town is kinda dodgy, you know, with the types who hang out there.",
        "She's decent-looking for her weight, but hitting the gym wouldn't hurt.",
        "Immigrants snag all the gigs, though the ones who blend in are alright I suppose.",
        "Same-sex stuff is whatever, but I'd rather my kids stick to the normal path.",
        "The sunset was beautiful, painting the sky in shades of orange and pink.",
        "I love trying new recipes; cooking brings people together.",
        "Exercise in the morning boosts energy for the whole day.",
        "Reading expands your mind and opens up new worlds.",
        "Kindness goes a long way in making someone's day better."
    ]

    for i, sentence in enumerate(sentences, start=1):
        topic_scores = get_group_scores(sentence)
        result = fuzzy_decision(topic_scores)

        print("=" * 80)
        print(f"Sentence {i}: {sentence}")
        print("Topic scores:", topic_scores)
        print("Levels:", result["levels"])
        print("Aggregated:", result["aggregated"])
        print("Final label:", result["final_label"])

