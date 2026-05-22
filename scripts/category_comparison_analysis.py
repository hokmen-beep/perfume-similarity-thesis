import os
import pandas as pd

from scipy.stats import kruskal, mannwhitneyu


base_path = "/Users/helinokmen/Desktop/thesis_coding_component"

# Loading the final pair dataset
df = pd.read_csv(
    os.path.join(base_path, "data", "final_master_dataset.csv")
)

# Checking the available category names
print("Category values in the file:")
print(df["category"].value_counts())


# These are the computational similarity features tested
features = [
    "embedding_minilm",
    "note",
    "tfidf"
]

# The three similarity categories used in the experiment
categories = [
    "high",
    "medium",
    "low/random"
]

all_results = []


# Running category comparison tests for each feature
for feature in features:

    print("\n" + "=" * 60)
    print("FEATURE:", feature)

    groups = []

    # Getting values for each category
    for category in categories:

        values = df[
            df["category"] == category
        ][feature]

        groups.append(values)

        print(category, "n =", len(values))

    # Overall comparison across all three categories
    h_stat, p_value = kruskal(*groups)

    print("Kruskal-Wallis H:", round(h_stat, 4))
    print("p-value:", p_value)

    all_results.append({
        "feature": feature,
        "test": "kruskal_wallis",
        "comparison": "high_medium_low_random",
        "statistic": h_stat,
        "p_value": p_value
    })

    # Pairwise comparisons between categories
    comparisons = [
        ("high", "medium"),
        ("high", "low/random"),
        ("medium", "low/random")
    ]

    print("\nPairwise Mann-Whitney U tests:")

    for a, b in comparisons:

        x = df[df["category"] == a][feature]
        y = df[df["category"] == b][feature]

        # Comparing two categories at a time
        u_stat, p_pair = mannwhitneyu(
            x,
            y,
            alternative="two-sided"
        )

        print(
            f"{a} vs {b}: "
            f"U={round(u_stat, 4)}, "
            f"p={p_pair}"
        )

        all_results.append({
            "feature": feature,
            "test": "mann_whitney_u",
            "comparison": f"{a}_vs_{b}",
            "statistic": u_stat,
            "p_value": p_pair
        })


# Saving all statistical test results
results = pd.DataFrame(all_results)

results.to_csv(
    os.path.join(base_path, "outputs", "effect_size_results.csv"),
    index=False
)

print("\nDone")
print("Saved effect_size_results.csv")