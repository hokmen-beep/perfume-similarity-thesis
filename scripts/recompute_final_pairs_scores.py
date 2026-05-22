import os
import re
import ast
import unicodedata
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


# Project path
base_path = "/Users/helinokmen/Desktop/thesis_coding_component"

# Input files
perfumes_file = os.path.join(base_path, "fragrantica_dataset", "perfumes_table.csv")
pairs_file = os.path.join(base_path, "data", "final_pairs.csv")
embeddings_file = os.path.join(base_path, "data", "desc_emb_minilm.npy")

# Output file
output_file = os.path.join(base_path, "data", "final_master_dataset.csv")


# Loading datasets
perfumes = pd.read_csv(perfumes_file)
pairs = pd.read_csv(pairs_file)

# Loading MiniLM sentence-transformer embeddings
emb_mini = np.load(embeddings_file)


# Cleaning empty values
def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


# Cleaning text to make comparison of descriptions and notes easier
def simple_text(x):
    x = clean_text(x).lower()

    x = unicodedata.normalize("NFKD", x)
    x = "".join(c for c in x if not unicodedata.combining(c))

    x = re.sub(r"[^a-z0-9\s]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()

    return x


# Converting perfume notes to a clean set
def get_notes(x):
    if pd.isna(x):
        return set()

    try:
        notes = ast.literal_eval(x)
        return {simple_text(note) for note in notes if clean_text(note)}
    except:
        return set()


# Calculating note overlap using Jaccard similarity
def note_similarity(notes_1, notes_2):
    if not notes_1 and not notes_2:
        return 0.0

    return len(notes_1 & notes_2) / len(notes_1 | notes_2)


# Cleaning columns
perfumes["title"] = perfumes["title"].apply(clean_text)
perfumes["description"] = perfumes["description"].apply(clean_text)
perfumes["notes_set"] = perfumes["notes"].apply(get_notes)
perfumes["desc_clean"] = perfumes["description"].apply(simple_text)

pairs["export_tag"] = pairs["export_tag"].astype(str).str.lower().str.strip()


# Keeping only perfumes with descriptions
has_description = perfumes["description"] != ""

perfumes = perfumes[has_description].reset_index(drop=True)
emb_mini = emb_mini[has_description.to_numpy()]


# Create dictionary to quickly find perfume rows
title_to_row = {}

for i, title in enumerate(perfumes["title"]):
    title_to_row[title] = i


# Creating TF-IDF vectors from perfume descriptions
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(perfumes["desc_clean"])


results = []

for _, pair in pairs.iterrows():

    perfume_1 = pair["perfume_1"]
    perfume_2 = pair["perfume_2"]

    # Skipping pair if perfume title cannot be found
    if perfume_1 not in title_to_row:
        print("Missing perfume:", perfume_1)
        continue

    if perfume_2 not in title_to_row:
        print("Missing perfume:", perfume_2)
        continue

    row_1 = title_to_row[perfume_1]
    row_2 = title_to_row[perfume_2]

    # Similarity based on MiniLM sentence embeddings
    score_minilm = cosine_similarity(
        [emb_mini[row_1]],
        [emb_mini[row_2]]
    )[0][0]

    # Similarity based on shared description words
    score_tfidf = cosine_similarity(
        tfidf_matrix[row_1],
        tfidf_matrix[row_2]
    )[0][0]

    # Similarity based on shared perfume notes
    score_notes = note_similarity(
        perfumes.loc[row_1, "notes_set"],
        perfumes.loc[row_2, "notes_set"]
    )

    # Simple score used during pair selection
    final_score = (score_minilm + score_notes + score_tfidf) / 3

    results.append({
        "export_tag": pair["export_tag"],
        "perfume_1": perfume_1,
        "perfume_2": perfume_2,
        "category": pair["category"],
        "block": pair["block"],
        "embedding_minilm": score_minilm,
        "note": score_notes,
        "tfidf": score_tfidf,
        "final_score": final_score
    })


# Convert results into a dataframe
output = pd.DataFrame(results)

# Save final dataset
output.to_csv(output_file, index=False)

print("Done")
print("Number of final pairs:", len(output))
print(output.head())
print("Saved final_master_dataset.csv")
