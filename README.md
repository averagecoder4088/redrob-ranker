# Redrob Hackathon — Intelligent Candidate Ranker

## Overview

This repository contains the ranking pipeline developed for the Redrob AI Hackathon.

The system retrieves the most relevant candidates using semantic search with precomputed BGE embeddings and FAISS, then reranks them using experience and behavioral signals before generating the final submission.

---

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Required Artifacts

Place the following files in the same directory as `rank.py`:

```
candidate_embeddings.npy
candidate_ids_ordered.npy
jd_embedding.npy
```

These artifacts are generated during the offline preprocessing stage and are required for ranking.

---

## Input

The ranking script expects:

```
candidates.jsonl
```

---

## Running the Ranker

```bash
python rank.py \
    --candidates ./candidates.jsonl \
    --out ./submission.csv \
    --artifacts .
```

The script will:

1. Load precomputed embeddings
2. Retrieve the Top-1000 candidates using FAISS cosine similarity
3. Compute experience and behavior scores
4. Produce the final Top-100 ranking
5. Generate deterministic reasoning strings
6. Validate the submission format
7. Write `submission.csv`

---

## Output

The generated submission contains the following columns:

```
candidate_id
rank
score
reasoning
```

---

## Runtime

- Runtime: ~3 seconds on CPU
- GPU required: No
- Internet access required: No
- Ranking uses only local artifacts

---

## Method Summary

The ranking score is computed as:

```
70% Semantic Similarity
15% Experience Score
15% Behavior Score
```

Semantic retrieval uses **BAAI/bge-base-en-v1.5** embeddings indexed with **FAISS**.

Experience scoring incorporates years of experience and employment stability.

Behavior scoring incorporates recruiter responsiveness, recent activity, notice period, GitHub activity, interview completion, and engagement signals.

Reasoning strings are deterministic, candidate-specific, and generated only from candidate profile facts.

---

## Reproducibility

The repository is fully reproducible.

Running `rank.py` with the required artifacts and the provided `candidates.jsonl` file reproduces the final submission without requiring internet access or GPU acceleration.