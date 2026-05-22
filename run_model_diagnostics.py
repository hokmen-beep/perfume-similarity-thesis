import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr


base_path = "/Users/helinokmen/Desktop/thesis_coding_component"

pairs = pd.read_csv(os.path.join(base_path, "data", "final_master_dataset.csv"))
human = pd.read_csv(os.path.join(base_path, "outputs", "pair_level_human_summary.csv"))

# Making sure the pair IDs match in both files
pairs["export_tag"] = pairs["export_tag"].str.lower().str.strip()
human["export_tag"] = human["export_tag"].str.lower().str.strip()

# Combining human ratings with computational similarity scores
data = human.merge(pairs, on="export_tag")

print("Matched pairs:", len(data))

y = data["mean_rating"]


# Runing a simple regression model
def test_model(x_columns, model_name):
    X = data[x_columns]

    model = LinearRegression()
    model.fit(X, y)

    predictions = model.predict(X)

    r2 = r2_score(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    spearman = spearmanr(y, predictions)

    return {
        "model": model_name,
        "r2": r2,
        "rmse": rmse,
        "spearman": spearman.statistic,
        "spearman_p": spearman.pvalue
    }


# Getting regression coefficients and p-values
def ols_model(x_columns, model_name):
    X = data[x_columns]
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    print("\nOLS results for:", model_name)
    print(model.summary())

    table = pd.DataFrame({
        "model": model_name,
        "term": model.params.index,
        "coefficient": model.params.values,
        "p_value": model.pvalues.values
    })

    return table


# This function adds a simple regression line to scatter plots
def add_regression_line(x_column):
    x_values = data[x_column].values
    y_values = data["mean_rating"].values

    slope, intercept = np.polyfit(x_values, y_values, 1)

    line_x = np.linspace(x_values.min(), x_values.max(), 100)
    line_y = slope * line_x + intercept

    plt.plot(line_x, line_y)


# Run the supporting regression models:
# 1. description embedding similarity only
# 2. TF-IDF description similarity only
# 3. fragrance note overlap only
# 4. description embedding similarity and note overlap together
models = [
    (["embedding_minilm"], "Description Embedding"),
    (["tfidf"], "TF-IDF"),
    (["note"], "Note"),
    (["embedding_minilm", "note"], "Combined")
]

results = []
ols_results = []

for columns, name in models:
    result = test_model(columns, name)
    results.append(result)

    ols_result = ols_model(columns, name)
    ols_results.append(ols_result)


model_results = pd.DataFrame(results)
ols_results = pd.concat(ols_results, ignore_index=True)

print("\nDiagnostic model results:")
print(model_results)

print("\nOLS coefficient results:")
print(ols_results)

model_results.to_csv(
    os.path.join(base_path, "outputs", "diagnostic_model_results.csv"),
    index=False
)

ols_results.to_csv(
    os.path.join(base_path, "outputs", "ols_results.csv"),
    index=False
)


# Adding predictions from the combined model for inspection
combined_model = LinearRegression()
combined_model.fit(data[["embedding_minilm", "note"]], y)

data["combined_prediction"] = combined_model.predict(
    data[["embedding_minilm", "note"]]
)

data.to_csv(os.path.join(base_path, "outputs", "prediction_table.csv"), index=False)


# Category summary: HS = high similarity, MS = medium similarity, RD = random/low similarity
data["category"] = data["export_tag"].str[:2]

category_summary = data.groupby("category")["mean_rating"].agg(["mean", "std", "count"])

print("\nCategory summary:")
print(category_summary)

category_summary.to_csv(os.path.join(base_path, "outputs", "category_summary.csv"))


# High note-overlap check
high_note_pairs = data[data["note"] >= 0.70][
    ["export_tag", "mean_rating", "embedding_minilm", "note"]
]

print("\nHigh note-overlap pairs:")
print(high_note_pairs)

high_note_pairs.to_csv(os.path.join(base_path, "outputs", "high_note_pairs.csv"), index=False)


# Plot 1: note overlap vs human ratings
plt.figure(figsize=(6, 5))
plt.scatter(data["note"], data["mean_rating"])
add_regression_line("note")
plt.xlabel("Note overlap")
plt.ylabel("Mean human rating")
plt.title("Human ratings vs note overlap")
plt.savefig(os.path.join(base_path, "outputs", "plot_note_vs_rating.png"))
plt.close()


# Plot 2: description embedding similarity vs human ratings
plt.figure(figsize=(6, 5))
plt.scatter(data["embedding_minilm"], data["mean_rating"])
add_regression_line("embedding_minilm")
plt.xlabel("Description embedding similarity")
plt.ylabel("Mean human rating")
plt.title("Human ratings vs description embedding similarity")
plt.savefig(os.path.join(base_path, "outputs", "plot_description_embedding_vs_rating.png"))
plt.close()


# Plot 3: TF-IDF description similarity vs human ratings
plt.figure(figsize=(6, 5))
plt.scatter(data["tfidf"], data["mean_rating"])
add_regression_line("tfidf")
plt.xlabel("TF-IDF description similarity")
plt.ylabel("Mean human rating")
plt.title("Human ratings vs TF-IDF description similarity")
plt.savefig(os.path.join(base_path, "outputs", "plot_tfidf_vs_rating.png"))
plt.close()


# Plot 4: average rating by category
plt.figure(figsize=(6, 5))
plt.bar(category_summary.index, category_summary["mean"])
plt.xlabel("Category")
plt.ylabel("Mean human rating")
plt.title("Mean ratings by category")
plt.savefig(os.path.join(base_path, "outputs", "plot_category_means.png"))
plt.close()

print("\nDiagnostic analysis finished. Output files are saved in the outputs folder.")

# Plot 5: correlation heatmap


correlation_data = data[["mean_rating", "embedding_minilm", "tfidf", "note"]].copy()
correlation_data.columns = ["Mean rating", "Description embedding", "TF-IDF", "Note overlap"]

plt.figure(figsize=(6, 5))

sns.heatmap(correlation_data.corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation heatmap")

plt.tight_layout()

plt.savefig( os.path.join(base_path, "outputs", "plot_correlation_heatmap2.png"), dpi=300)

plt.close()

# Plot 6: residual plot for the combined model
residuals = y - data["combined_prediction"]

plt.figure(figsize=(6, 5))

plt.scatter(data["combined_prediction"], residuals)

plt.axhline(0)

plt.xlabel("Predicted rating")
plt.ylabel("Residual")
plt.title("Residual plot for combined model")

plt.savefig(
    os.path.join(base_path, "outputs", "plot_residuals_combined.png")
)

plt.close()
