"""
rank.py — Redrob Hackathon Candidate Ranker

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv --artifacts .

Artifacts required in --artifacts directory:
    candidate_embeddings.npy
    candidate_ids_ordered.npy
    jd_embedding.npy
"""

import argparse
import hashlib
import json
import time
from datetime import date
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

TODAY = date(2026, 6, 19)

# ── SCORING ───────────────────────────────────────────────────────────────────
def compute_avg_tenure_months(career_history):
    company_durations = {}
    for job in career_history:
        company = job.get('company', 'unknown').lower().strip()
        months  = job.get('duration_months', 0)
        company_durations[company] = company_durations.get(company, 0) + months
    if not company_durations:
        return None
    return sum(company_durations.values()) / len(company_durations)

def title_chaser_multiplier(candidate):
    years = candidate['profile']['years_of_experience']
    if years < 5:
        return 1.00
    avg = compute_avg_tenure_months(candidate['career_history'])
    if avg is None:
        return 1.00
    if years <= 9:
        if avg >= 30:  return 1.00
        elif avg >= 24: return 0.95
        elif avg >= 18: return 0.90
        else:           return 0.70
    else:
        if avg >= 36:  return 1.00
        elif avg >= 24: return 0.90
        elif avg >= 18: return 0.80
        else:           return 0.60

def years_score(y):
    if y < 2:    return 0
    elif y < 3:  return 30
    elif y < 4:  return 60
    elif y < 5:  return 80
    elif y < 6:  return 95
    elif y < 8:  return 100
    elif y < 9:  return 95
    elif y < 10: return 92
    elif y < 12: return 90
    elif y < 15: return 85
    else:        return 80

def stability_score(mult):
    return {1.00: 100, 0.95: 95, 0.90: 90, 0.80: 80, 0.70: 70}.get(round(mult, 2), 100)

def experience_score(years, mult):
    return 0.60 * years_score(years) + 0.40 * stability_score(mult)

def behavior_score(signals):
    last_active = date.fromisoformat(signals['last_active_date'])
    days = (TODAY - last_active).days

    rr  = signals['recruiter_response_rate']
    rr_s = (0   if rr < 0.10 else
            40  if rr < 0.30 else
            70  if rr < 0.50 else
            85  if rr < 0.70 else
            95  if rr < 0.90 else 100)

    la_s = (100 if days <= 30  else
            85  if days <= 90  else
            60  if days <= 180 else 20)

    np_  = signals['notice_period_days']
    np_s = (100 if np_ <= 30 else
            90  if np_ <= 60 else
            80  if np_ <= 90 else 65)

    ow_s = 100 if signals['open_to_work_flag'] else 90

    gh   = signals['github_activity_score']
    gh_s = (30  if gh == -1 else
            50  if gh < 25  else
            70  if gh < 50  else
            85  if gh < 75  else 100)

    rt   = signals['avg_response_time_hours']
    rt_s = (100 if rt <= 6   else
            90  if rt <= 24  else
            75  if rt <= 72  else
            50  if rt <= 168 else 20)

    ic   = signals['interview_completion_rate']
    ic_s = (30  if ic < 0.50 else
            70  if ic < 0.75 else
            90  if ic < 0.90 else 100)

    ap   = signals['applications_submitted_30d']
    ap_s = 50 if ap == 0 else 80 if ap <= 5 else 100

    t1 = 0.35 * rr_s + 0.25 * la_s + 0.25 * np_s + 0.15 * ow_s
    t2 = 0.40 * gh_s + 0.25 * rt_s + 0.20 * ic_s + 0.15 * ap_s
    return 0.75 * t1 + 0.25 * t2

# ── REASONING ─────────────────────────────────────────────────────────────────
SEARCH_RETRIEVAL_TITLES = {
    'search engineer', 'recommendation systems engineer', 'nlp engineer',
    'senior nlp engineer', 'ml engineer', 'senior ml engineer',
    'machine learning engineer', 'senior machine learning engineer',
    'staff machine learning engineer', 'ai engineer', 'senior ai engineer',
    'lead ai engineer', 'applied ml engineer', 'senior applied scientist',
    'ai research engineer', 'junior ml engineer'
}

def get_variant(candidate_id):
    return int(hashlib.md5(candidate_id.encode()).hexdigest(), 16) % 3

def get_top_skills(candidate):
    return [
        s['name'] for s in sorted(
            candidate['skills'],
            key=lambda s: s['duration_months'],
            reverse=True
        )[:3]
    ]

def generate_reasoning(candidate):
    profile = candidate['profile']
    signals = candidate['redrob_signals']
    cid     = candidate['candidate_id']

    title  = profile.get('current_title') or profile.get('headline') or 'AI Professional'
    years  = profile.get('years_of_experience', 0)
    skills = get_top_skills(candidate)
    skills_str = ', '.join(skills) if skills else None

    appenders = []
    if signals.get('open_to_work_flag'):
        appenders.append('open to work')
    if signals.get('notice_period_days', 999) <= 60:
        appenders.append(f"notice period {signals['notice_period_days']} days")
    if signals.get('recruiter_response_rate', 0) >= 0.50:
        appenders.append(f"recruiter response rate {signals['recruiter_response_rate']:.2f}")

    career_context = None
    if title.lower() in SEARCH_RETRIEVAL_TITLES:
        career_context = "Background includes search, retrieval or ranking-related roles"

    variant = get_variant(cid)
    if variant == 0:
        core = f"{title} with {years:.1f} years of experience"
        if skills_str: core += f". Top skills include {skills_str}."
    elif variant == 1:
        core = f"{years:.1f} years of experience as {title}"
        if skills_str: core += f". Key skills include {skills_str}."
    else:
        core = f"{title} with {years:.1f} years of experience"
        if skills_str: core += f". Skills: {skills_str}."

    if career_context:
        core += f" {career_context}."

    if appenders:
        appenders[0] = appenders[0].capitalize()
        core += ' ' + '; '.join(appenders) + '.'

    return core

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Redrob candidate ranker')
    parser.add_argument('--candidates', required=True, help='Path to candidates.jsonl')
    parser.add_argument('--out',        required=True, help='Output CSV path')
    parser.add_argument('--artifacts',  default='.',   help='Directory containing .npy files')
    args = parser.parse_args()

    t0       = time.time()
    artifacts = Path(args.artifacts)

    print("Loading artifacts...")
    embeddings  = np.load(artifacts / 'candidate_embeddings.npy', allow_pickle=True).astype('float32')
    ids_ordered = np.load(artifacts / 'candidate_ids_ordered.npy', allow_pickle=True)
    jd_emb      = np.load(artifacts / 'jd_embedding.npy', allow_pickle=True).astype('float32')
    print(f"  embeddings: {embeddings.shape}  ({time.time()-t0:.1f}s)")

    print("Running FAISS search...")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    scores, indices = index.search(jd_emb, k=1000)
    top1000_ids = set(ids_ordered[indices[0]])
    cosine_map  = {ids_ordered[idx]: float(score)
                   for idx, score in zip(indices[0], scores[0])}
    print(f"  top-1000 retrieved  ({time.time()-t0:.1f}s)")

    print("Loading candidate profiles...")
    candidate_lookup = {}
    with open(args.candidates) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            if c['candidate_id'] in top1000_ids:
                candidate_lookup[c['candidate_id']] = c
    print(f"  loaded {len(candidate_lookup)} profiles  ({time.time()-t0:.1f}s)")

    print("Scoring...")
    rows = []
    for cid, cosine in cosine_map.items():
        c    = candidate_lookup[cid]
        yrs  = c['profile']['years_of_experience']
        mult = title_chaser_multiplier(c)
        exp  = experience_score(yrs, mult)
        beh  = behavior_score(c['redrob_signals'])
        base = 0.70 * (cosine * 100) + 0.15 * exp + 0.15 * beh
        rows.append({'candidate_id': cid, 'base_score': base, 'candidate_obj': c})

    df = pd.DataFrame(rows)

    top100 = (
        df.sort_values(by=['base_score', 'candidate_id'], ascending=[False, True])
        .head(100)
        .reset_index(drop=True)
    )
    top100['rank'] = top100.index + 1

    print("Generating reasoning...")
    reasoning_strings = [
        generate_reasoning(row['candidate_obj'])
        for _, row in top100.iterrows()
    ]

    submission = pd.DataFrame({
        'candidate_id': top100['candidate_id'].values,
        'rank':         top100['rank'].values,
        'score':        (top100['base_score'] / 100).round(4).values,
        'reasoning':    reasoning_strings,
    })

    submission['score'] = submission['score'].astype(float).round(4)
    submission = (
        submission
        .sort_values(by=['score', 'candidate_id'], ascending=[False, True])
        .reset_index(drop=True)
    )
    submission['rank'] = submission.index + 1

    assert len(submission) == 100
    assert submission['candidate_id'].is_unique
    assert submission['rank'].tolist() == list(range(1, 101))
    assert (submission['score'].diff().dropna() <= 0).all()
    assert (submission['reasoning'].fillna('').str.strip().ne('')).all()
    print("All validator checks passed.")

    submission.to_csv(args.out, index=False)
    print(f"\nDone in {time.time()-t0:.1f}s")
    print(f"Saved to: {args.out}")
    print(submission[['rank','candidate_id','score','reasoning']].head(5).to_string(index=False))

if __name__ == '__main__':
    main()