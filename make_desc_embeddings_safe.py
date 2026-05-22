import os
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer


# Main folder path
base_path = "/Users/helinokmen/Desktop/thesis_coding_component"

# Input and output files
input_file = os.path.join(base_path, "fragrantica_dataset", "perfumes_table.csv")

output_embeddings = os.path.join(base_path, "data", "desc_emb_minilm.npy")
output_metadata = os.path.join(base_path, "data", "desc_emb_minilm_meta.csv")


# Loading perfume dataset
df = pd.read_csv(input_file)

# Get perfume descriptions
# Empty descriptions are replaced with empty strings
texts = df["description"].fillna("").astype(str).tolist()


# Loading MiniLM sentence-transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Converting descriptions into embeddings
embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

print("Embeddings shape:", embeddings.shape)


# Save embeddings as a NumPy file
np.save(output_embeddings, embeddings)


# Save some metadata for easier inspection later
df[["title", "url", "designer", "rating", "notes"]].to_csv(
    output_metadata,
    index=False
)

print("Done")
print("Saved embeddings:", output_embeddings)
print("Saved metadata:", output_metadata)
