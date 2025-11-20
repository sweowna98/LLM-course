from datasets import load_dataset
import pandas as pd
import tqdm
from sentence_transformers import SentenceTransformer
import faiss

# Dataset preprocessing and conversion to RAG

df = pd.read_csv("Travel details dataset.csv")
df['Traveler age'] = df['Traveler age'].fillna(0).astype(int)
df['Duration (days)'] = df['Duration (days)'].fillna(0).astype(int)
df['Accommodation cost'] = df['Accommodation cost'].astype(str)
df['Transportation cost'] = df['Transportation cost'].astype(str)
df['Traveler gender'] = df['Traveler gender'].str.lower()


def row_to_sentence(row):
    return (
        f"({row['Traveler gender']}, {row['Traveler age']} years old, " #{row['Traveler name']} removed
        f"{row['Traveler nationality']}) went to {row['Destination']} from {row['Start date']} to {row['End date']} "
        f"({row['Duration (days)']} days). They stayed at a {row['Accommodation type']} costing ${row['Accommodation cost']}, "
        f"and used {row['Transportation type']} costing ${row['Transportation cost']}."
    )

df['RowString'] = df.apply(row_to_sentence, axis=1)
df_mistral = df['RowString']

print(df_mistral)



# Generate embeddings + FAISS index
embedder = SentenceTransformer("all-MiniLM-L6-v2")

texts = df['RowString'].tolist()
embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True).astype("float32")

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension) 
index.add(embeddings)
print(f"Indexed {index.ntotal} rows")