# Redrob Hackathon — Candidate Ranker

## Setup
pip install -r requirements.txt

## Artifacts required
Place these files in the same directory as rank.py:
- candidate_embeddings.npy
- candidate_ids_ordered.npy
- jd_embedding.npy

## Reproduce submission
python rank.py --candidates ./candidates.jsonl --out ./submission.csv

## Runtime
~3 seconds on CPU. No GPU required. No network calls during ranking.
