import os
import pandas as pd
from scipy.stats import spearmanr, kruskal


base_path = "/Users/helinokmen/Desktop/thesis_coding_component"

input_file = os.path.join(base_path, "outputs", "prediction_table.csv")
output_folder = os.path.join(base_path, "outputs")

data = pd.read_csv(input_file)

# Creating category labels from the pair names
data["category"] = data["export_tag"].str[:2]

# Checking how many pairs are in each category
category_counts = data["category"].value_counts()
print("\nCategory counts:")
print(category_counts)

# Checking the average scores for each category
category_summary = data.groupby("category")[
    ["mean_rating", "embedding_minilm", "tfidf", "note"]
].mean().round(3)

print("\nCategory summary:")
print(category_summary)

category_summary.to_csv(
    os.path.join(output_folder, "validation_category_summary.csv")
)

# Checking correlations with human ratings
results = []

for column in ["embedding_minilm", "tfidf", "note"]:
    correlation = spearmanr(data["mean_rating"], data[column])

    results.append({
        "variable": column,
        "spearman_r": correlation.statistic,
        "p_value": correlation.pvalue
    })

correlation_results = pd.DataFrame(results)

print("\nCorrelations with human ratings:")
print(correlation_results)

correlation_results.to_csv(
    os.path.join(output_folder, "validation_correlations.csv"),
    index=False
)

# Checking ig the human ratings differ between High similarity , Medium similarity , and Random similarity categories
hs = data[data["category"] == "hs"]["mean_rating"]
ms = data[data["category"] == "ms"]["mean_rating"]
rd = data[data["category"] == "rd"]["mean_rating"]

test = kruskal(hs, ms, rd)

print("\nKruskal-Wallis test:")
print("H:", test.statistic)
print("p:", test.pvalue)

print("\nValidation finished.")
