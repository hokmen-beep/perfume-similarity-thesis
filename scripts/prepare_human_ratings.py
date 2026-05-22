import os
import pandas as pd

base_path = "/Users/helinokmen/Desktop/thesis_coding_component"

# paths
input_file = os.path.join(base_path, "data", "qualtrics_collected_data.csv")
output_folder = os.path.join(base_path, "outputs")

# Loading data
df = pd.read_csv(input_file)

# Removeing non-response rows from Qualtrics export
df = df[df["ResponseId"].notna()].copy()

# Finding perfume-pair rating columns
pair_columns = [col for col in df.columns if col.startswith(("HS", "MS", "RD"))]

# Keeping only participants who passed the attention check
if "ATTNCHK" in df.columns:
    df["ATTNCHK"] = df["ATTNCHK"].astype(str).str.strip()
    df = df[(df["ATTNCHK"] == "5") | (df["ATTNCHK"].str.lower() == "strongly agree")]

# Keeping only participants who answered all perfume-pair ratings
df[pair_columns] = df[pair_columns].apply(pd.to_numeric, errors="coerce")
df = df.dropna(subset=pair_columns)

df["participant_id"] = range(1, len(df) + 1)

# Converting Qualtrics data into long format
ratings = df.melt(
    id_vars="participant_id",
    value_vars=pair_columns,
    var_name="export_tag",
    value_name="rating"
)

# Cleaning ratings
ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
ratings = ratings.dropna(subset=["rating"])

# Cleaning pair names
ratings["export_tag"] = ratings["export_tag"].str.lower().str.strip()

# Fixing pair name differences so they match final_master_dataset.csv
ratings["export_tag"] = ratings["export_tag"].replace({
    "hs2_eternalkis_honey": "hs2_etkiss_honey",
    "ms2_onemillion_lemal": "ms2_1mil_lemale",
    "hs4_burberyher_scand": "hs4_burbher_scand"
})

# Creating a pair-level summary
pair_summary = ratings.groupby("export_tag").agg(
    mean_rating=("rating", "mean"),
    sd_rating=("rating", "std"),
    n=("rating", "count")
).reset_index()

# Saving outputs
ratings.to_csv(os.path.join(output_folder, "human_ratings_long.csv"), index=False)
pair_summary.to_csv(os.path.join(output_folder, "pair_level_human_summary.csv"), index=False)

print("Done")
print("Valid participants:", df["participant_id"].nunique())
print("Number of perfume pairs:", len(pair_columns))
print("Number of pair ratings:", len(ratings))
print(pair_summary)
