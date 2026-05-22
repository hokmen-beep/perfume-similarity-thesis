import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import LeaveOneOut
from scipy.stats import spearmanr


base_path = "/Users/helinokmen/Desktop/thesis_coding_component"

pairs_file = os.path.join(base_path, "data", "final_master_dataset.csv")
human_file = os.path.join(base_path, "outputs", "pair_level_human_summary.csv")
output_folder = os.path.join(base_path, "outputs")

# Loading the computational pair scores and the human rating summary
pairs = pd.read_csv(pairs_file)
human = pd.read_csv(human_file)

# Cleaning pair names so the two files can be matched correctly
pairs["export_tag"] = pairs["export_tag"].str.lower().str.strip()
human["export_tag"] = human["export_tag"].str.lower().str.strip()

# Combining human ratings with the computational features
data = human.merge(pairs, on="export_tag")

print("Matched pairs:", len(data))

# the value the models try to predict
y = data["mean_rating"]

# prediction models tested in the thesis
models = [
    (["embedding_minilm"], "description_embedding"),
    (["tfidf"], "tfidf"),
    (["note"], "note"),
    (["embedding_minilm", "note"], "combined")
]


# Creating a function to predict human ratings with leave-one-out cross-validation
def run_prediction_model(columns, model_name):
    X = data[columns]

    loo = LeaveOneOut()
    rows = []

    # Each time, one perfume pair is left out as the test pair
    for train_i, test_i in loo.split(X):
        X_train = X.iloc[train_i]
        X_test = X.iloc[test_i]

        y_train = y.iloc[train_i]
        y_test = y.iloc[test_i]

        # Training the model on the remaining perfume pairs
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predicting the human rating for the left-out perfume pair
        predicted = model.predict(X_test)[0]

        rows.append({
            "model": model_name,
            "export_tag": data.iloc[test_i[0]]["export_tag"],
            "actual_rating": y_test.iloc[0],
            "predicted_rating": predicted
        })

    return pd.DataFrame(rows)


# Running prediction for all models
all_predictions = []

for columns, model_name in models:
    predictions = run_prediction_model(columns, model_name)
    all_predictions.append(predictions)

all_predictions = pd.concat(all_predictions, ignore_index=True)


# Calculating prediction performance for each model
results = []

for model_name in all_predictions["model"].unique():
    model_data = all_predictions[all_predictions["model"] == model_name]

    actual = model_data["actual_rating"]
    predicted = model_data["predicted_rating"]

    spearman = spearmanr(actual, predicted)

    results.append({
        "model": model_name,
        "r2": r2_score(actual, predicted),
        "rmse": np.sqrt(mean_squared_error(actual, predicted)),
        "spearman": spearman.statistic,
        "spearman_p": spearman.pvalue
    })

results = pd.DataFrame(results)

print("\nPrediction results:")
print(results)

# Saving prediction outputs
all_predictions.to_csv(
    os.path.join(output_folder, "loocv_predictions.csv"),
    index=False
)

results.to_csv(
    os.path.join(output_folder, "prediction_model_results.csv"),
    index=False
)


# Creating actual vs predicted plots for each model
for model_name in all_predictions["model"].unique():
    model_data = all_predictions[all_predictions["model"] == model_name]

    actual = model_data["actual_rating"]
    predicted = model_data["predicted_rating"]

    plt.figure(figsize=(6, 5))
    plt.scatter(actual, predicted)

    # The diagonal line shows perfect prediction
    line_min = min(actual.min(), predicted.min())
    line_max = max(actual.max(), predicted.max())

    plt.plot([line_min, line_max], [line_min, line_max])

    plt.xlabel("Actual human rating")
    plt.ylabel("Predicted human rating")
    plt.title("Actual vs predicted: " + model_name)

    plt.savefig(
        os.path.join(output_folder, "actual_vs_predicted_" + model_name + ".png")
    )

    plt.close()


print("\nPrediction analysis finished.")
print("Output files were saved in the outputs folder.")
