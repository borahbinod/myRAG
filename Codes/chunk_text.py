# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:22:30 2026

@author: binod
"""

from pathlib import Path
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PARSED_DIR = BASE_DIR / "parsed"

# ============================================================
# Load parsed pages
# ============================================================

pages = pd.read_pickle(PARSED_DIR / "pages.pkl")

# ============================================================
# Chunking parameters
# ============================================================

CHUNK_SIZE = 500      # tokens (approximate)
CHUNK_OVERLAP = 75

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(

    chunk_size=CHUNK_SIZE,

    chunk_overlap=CHUNK_OVERLAP,

    separators=[

        "\n\n",
        "\n",
        ". ",
        "; ",
        ", ",
        " "

    ]

)

# ============================================================
# Chunk each paper
# ============================================================

chunks = []

chunk_id = 1

for paper_id, group in pages.groupby("paper_id"):

    group = group.sort_values("page")

    # Join all pages into one document
    document = "\n\n".join(group["text"].tolist())

    # Split into chunks
    split_text = splitter.split_text(document)

    # Approximate page mapping
    cumulative = ""
    page_lookup = []

    for _, row in group.iterrows():

        cumulative += row["text"] + "\n\n"

        page_lookup.append((len(cumulative), row["page"]))

    # Store chunks
    running = 0

    for text in split_text:

        start = running
        end = running + len(text)

        running = end

        page_start = None
        page_end = None

        for position, page in page_lookup:

            if page_start is None and start <= position:
                page_start = page

            if end <= position:
                page_end = page
                break

        if page_end is None:
            page_end = group["page"].max()

        chunks.append({

            "chunk_id": chunk_id,

            "paper_id": paper_id,

            "paper": group.iloc[0]["paper"],

            "page_start": page_start,

            "page_end": page_end,

            "text": text,

            "n_words": len(text.split()),

            "n_characters": len(text)

        })

        chunk_id += 1

# ============================================================
# Save
# ============================================================

chunks = pd.DataFrame(chunks)

chunks.to_pickle(PARSED_DIR / "chunks.pkl")
chunks.to_csv(PARSED_DIR / "chunks.csv", index=False)

print(chunks.head())

print(f"\nCreated {len(chunks)} chunks.")

