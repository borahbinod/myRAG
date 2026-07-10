# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 09:06:40 2026

@author: binod
"""
# ============================================================
# Project folders
# ============================================================

from pathlib import Path
import fitz
import pandas as pd
import re

# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

LITERATURE_DIR = BASE_DIR / "Literature"
OUTPUT_DIR = BASE_DIR / "parsed"

OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# Text cleaning
# ============================================================

def clean_text(text):

    # Fix common ligatures
    text = (text
            .replace("ﬁ", "fi")
            .replace("ﬂ", "fl")
            .replace("ﬀ", "ff")
            .replace("ﬃ", "ffi")
            .replace("ﬄ", "ffl"))

    # Remove hyphenation across lines
    text = re.sub(r"-\s*\n\s*", "", text)

    # Replace multiple newlines with one
    text = re.sub(r"\n{2,}", "\n\n", text)

    # Replace single newlines with spaces
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Collapse spaces
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()

# ============================================================
# Storage
# ============================================================

paper_rows = []
page_rows = []

pdf_files = sorted(LITERATURE_DIR.glob("*.pdf"))

print(f"\nFound {len(pdf_files)} PDFs\n")

# ============================================================
# Parse every PDF
# ============================================================

for paper_id, pdf in enumerate(pdf_files, start=1):

    print(f"Reading {pdf.name}")

    doc = fitz.open(pdf)

    meta = doc.metadata

    # -----------------------------
    # Paper metadata
    # -----------------------------

    paper_rows.append({

        "paper_id": paper_id,

        "filename": pdf.name,

        "paper": pdf.stem,

        "title": meta.get("title", ""),

        "authors": meta.get("author", ""),

        "subject": meta.get("subject", ""),

        "keywords": meta.get("keywords", ""),

        "creator": meta.get("creator", ""),

        "producer": meta.get("producer", ""),

        "creation_date": meta.get("creationDate", ""),

        "modification_date": meta.get("modDate", ""),

        "n_pages": len(doc)

    })

    # -----------------------------
    # Page text
    # -----------------------------

    for page_num, page in enumerate(doc, start=1):

        blocks = page.get_text("blocks")

        # sort by reading order
        blocks = sorted(blocks, key=lambda b: (b[1], b[0]))

        page_text = []

        for block in blocks:

            text = block[4].strip()

            if len(text) < 20:
                continue

            text = clean_text(text)

            page_text.append(text)

        page_text = "\n\n".join(page_text)

        page_rows.append({

            "paper_id": paper_id,

            "paper": pdf.stem,

            "page": page_num,

            "text": page_text,

            "n_words": len(page_text.split()),

            "n_characters": len(page_text)

        })

# ============================================================
# DataFrames
# ============================================================

papers = pd.DataFrame(paper_rows)
pages = pd.DataFrame(page_rows)

# ============================================================
# Save
# ============================================================

papers.to_pickle(OUTPUT_DIR / "papers.pkl")
papers.to_csv(OUTPUT_DIR / "papers.csv", index=False)

pages.to_pickle(OUTPUT_DIR / "pages.pkl")
pages.to_csv(OUTPUT_DIR / "pages.csv", index=False)

print("\nDone!")

print(f"Papers: {len(papers)}")
print(f"Pages : {len(pages)}")




