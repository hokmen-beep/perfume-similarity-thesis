import os
import ast
import re
import unicodedata
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


base_path = "/Users/helinokmen/Desktop/thesis_coding_component"

perfumes_file = os.path.join(base_path, "fragrantica_dataset", "perfumes_table.csv")
pairs_file = os.path.join(base_path, "data", "final_pairs.csv")
embeddings_file = os.path.join(base_path, "data", "desc_emb_minilm.npy")

output_file = os.path.join(base_path, "outputs", "candidate_pairs_simple.csv")


perfumes = pd.read_csv(perfumes_file)
pairs = pd.read_csv(pairs_file)
embeddings = np.load(embeddings_file)


# This handles empty values
def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


# This cleans text and removes formatting differences
def normalize_text(x):

    x = clean_text(x).lower()

    x = x.replace("’", "'")
    x = x.replace("`", "'")

    x = unicodedata.normalize("NFKD", x)
    x = "".join(c for c in x if not unicodedata.combining(c))

    x = x.replace("for women and men", " ")
    x = x.replace("for men and women", " ")
    x = x.replace("for women", " ")
    x = x.replace("for men", " ")

    x = x.replace("perfumes and colognes", " ")
    x = x.replace("perfume and skin", " ")

    x = re.sub(r"[^a-z0-9\s]", " ", x)
    x = re.sub(r"\s+", " ", x)

    return x.strip()


# This turns the notes column into a cleaned note set
def get_notes(x):

    if pd.isna(x):
        return set()

    try:
        notes = ast.literal_eval(x)

        return {
            normalize_text(note)
            for note in notes
            if clean_text(note) != ""
        }

    except:
        return set()


# This calculates note overlap between two perfumes
def note_jaccard(notes_1, notes_2):

    if len(notes_1) == 0 or len(notes_2) == 0:
        return 0

    return len(notes_1 & notes_2) / len(notes_1 | notes_2)


# Clean important columns
perfumes["title"] = perfumes["title"].apply(clean_text)
perfumes["designer"] = perfumes["designer"].apply(clean_text)
perfumes["description"] = perfumes["description"].apply(clean_text)

perfumes["description_clean"] = (
    perfumes["description"].apply(normalize_text)
)

perfumes["notes_set"] = (
    perfumes["notes"].apply(get_notes)
)


# Keeping only perfumes with descriptions
has_description = perfumes["description"] != ""

perfumes = perfumes[has_description].reset_index(drop=True)
embeddings = embeddings[has_description.to_numpy()]


# Creating TF-IDF vectors from cleaned descriptions
tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=6000
)

tfidf_matrix = tfidf.fit_transform(
    perfumes["description_clean"]
)


# Use selected seed perfumes to generate candidate pairs for manual inspection
seed_perfumes = (
    pairs["perfume_1"]
    .drop_duplicates()
    .tolist()
)


all_candidates = []

for seed in seed_perfumes:

    seed_rows = perfumes[
        perfumes["title"] == seed
    ]

    if len(seed_rows) == 0:
        print("Seed not found:", seed)
        continue

    seed_index = seed_rows.index[0]

    seed_brand = perfumes.loc[seed_index, "designer"]
    seed_notes = perfumes.loc[seed_index, "notes_set"]

    embedding_scores = cosine_similarity(
        embeddings[seed_index].reshape(1, -1),
        embeddings
    )[0]

    tfidf_scores = cosine_similarity(
        tfidf_matrix[seed_index],
        tfidf_matrix
    )[0]

    candidates = []

    for i in range(len(perfumes)):

        if i == seed_index:
            continue

        candidate_title = perfumes.loc[i, "title"]
        candidate_brand = perfumes.loc[i, "designer"]

        # Skip same-brand perfumes
        if normalize_text(seed_brand) == normalize_text(candidate_brand):
            continue

        # Skip perfumes with obvious name overlap
        seed_first_word = normalize_text(seed).split()[0]

        if seed_first_word in normalize_text(candidate_title):
            continue

        embedding_score = embedding_scores[i]
        tfidf_score = tfidf_scores[i]

        note_score = note_jaccard(
            seed_notes,
            perfumes.loc[i, "notes_set"]
        )

        # Keep only somewhat similar candidates
        if embedding_score < 0.50:
            continue

        final_score = (
            embedding_score +
            tfidf_score +
            note_score
        ) / 3

        candidates.append({
            "seed_perfume": seed,
            "candidate_perfume": candidate_title,
            "candidate_brand": candidate_brand,
            "embedding_minilm": embedding_score,
            "tfidf": tfidf_score,
            "note": note_score,
            "final_score": final_score
        })

    candidates = pd.DataFrame(candidates)

    if len(candidates) == 0:
        continue

    candidates = candidates.sort_values(
        "final_score",
        ascending=False
    ).head(20)

    all_candidates.append(candidates)


all_candidates = pd.concat(
    all_candidates,
    ignore_index=True
)

all_candidates.to_csv(
    output_file,
    index=False
)

print("Done")
print("Saved candidate_pairs_simple.csv")
