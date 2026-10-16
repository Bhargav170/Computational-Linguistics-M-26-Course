import pandas as pd

scores = pd.read_csv("scores.csv")
ratings = pd.read_csv("ratings.csv")

df = scores.merge(ratings, on="sent_idx")

columns = [
    "tri_logP",
    "tri_avgLogP",
    "tri_SLOR",
    "bi_logP",
    "bi_avgLogP",
    "bi_SLOR"
]

for col in columns:
    r = df[col].corr(df["rating"])
    print(f"{col}: {r:.4f}")