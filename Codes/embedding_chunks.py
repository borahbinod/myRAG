# -*- coding: utf-8 -*-
"""
Create embeddings and FAISS index for RAG retrieval

@author: binod
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import faiss
import pickle


# ==========================================================
# File locations
# ==========================================================

# Change this if needed
BASE_DIR = Path.cwd().parent

PARSED_DIR = BASE_DIR / "parsed"

PARSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


chunk_file = PARSED_DIR / "chunks.csv"

embedding_file = PARSED_DIR / "chunk_embeddings.npy"
index_file = PARSED_DIR / "chunk_index.faiss"
metadata_file = PARSED_DIR / "chunk_metadata.pkl"


# ==========================================================
# Load chunks
# ==========================================================

print("Loading chunks...")

chunks = pd.read_csv(chunk_file)


print("Initial chunks:", len(chunks))

print("\nColumns:")
print(chunks.columns.tolist())


# ==========================================================
# Clean text
# ==========================================================

chunks = (
    chunks
    .dropna(subset=["text"])
    .copy()
)


chunks["text"] = (
    chunks["text"]
    .astype(str)
    .str.replace("\n", " ", regex=False)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)


# remove very short chunks

chunks = chunks[
    chunks["text"].str.len() > 100
]


chunks.reset_index(
    drop=True,
    inplace=True
)


print(
    "Chunks after cleaning:",
    len(chunks)
)


# ==========================================================
# Create embedding text
# ==========================================================

def make_embedding_text(row):

    title = row["title"] if "title" in row else ""

    section = row["section"] if "section" in row else ""

    return f"""
Title:
{title}

Section:
{section}

Text:
{row['text']}
"""


chunks["embedding_text"] = chunks.apply(
    make_embedding_text,
    axis=1
)


# ==========================================================
# Load embedding model
# ==========================================================

print("Loading embedding model...")


model = SentenceTransformer(
    "sentence-transformers/all-mpnet-base-v2"
)


model.max_seq_length = 512


# ==========================================================
# Generate embeddings
# ==========================================================

print("Generating embeddings...")


embeddings = model.encode(
    chunks["embedding_text"].tolist(),

    batch_size=32,

    show_progress_bar=True,

    convert_to_numpy=True,

    normalize_embeddings=True
)


print(
    "Embedding matrix:",
    embeddings.shape
)


# ==========================================================
# Save embeddings
# ==========================================================

np.save(
    embedding_file,
    embeddings
)


print(
    "Saved:",
    embedding_file
)


# ==========================================================
# Create FAISS index
# ==========================================================

print("Creating FAISS index...")


dimension = embeddings.shape[1]


# Inner product on normalized vectors = cosine similarity

index = faiss.IndexFlatIP(
    dimension
)


index.add(
    embeddings
)


print(
    "FAISS vectors:",
    index.ntotal
)


faiss.write_index(
    index,
    str(index_file)
)


print(
    "Saved:",
    index_file
)


# ==========================================================
# Save metadata
# ==========================================================

metadata_columns = [
    "chunk_id",
    "paper_id",
    "title",
    "section",
    "text"
]


# keep only columns that exist

metadata_columns = [
    col for col in metadata_columns
    if col in chunks.columns
]


metadata = chunks[
    metadata_columns
]


with open(
    metadata_file,
    "wb"
) as f:

    pickle.dump(
        metadata,
        f
    )


print(
    "Saved:",
    metadata_file
)


print("\nEmbedding pipeline complete!")