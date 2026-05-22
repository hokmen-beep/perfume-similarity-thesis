import os
import pandas as pd
import matplotlib.pyplot as plt

# Folder paths
base_path = "/Users/helinokmen/Desktop/thesis_coding_component"

input_file = os.path.join( base_path,"outputs", "prediction_model_results.csv")

output_file = os.path.join(base_path,"outputs","plot_model_performance.png")

# Loading prediction results
df = pd.read_csv(input_file)


name_map = {
    "description_embedding": "Description\nEmbedding",
    "tfidf": "TF-IDF",
    "note": "Note\nOverlap",
    "combined": "Combined"
}

df["model_clean"] = df["model"].map(name_map)

# Creating the  figure
plt.figure(figsize=(7, 5))

plt.bar( df["model_clean"], df["spearman"])

# Labels and title
plt.ylabel("Spearman correlation")
plt.xlabel("Prediction model")

plt.title(
    "Prediction performance across computational models",
    pad=12)


plt.tight_layout()


plt.savefig( output_file, dpi=300, bbox_inches="tight")

plt.close()

print("Saved plot_model_performance.png")
