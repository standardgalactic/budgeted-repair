"""Convention survival when newcomers inherit a budgeted slice of a lexicon.

Same game and turnover scheme as turnover.py. Each newcomer is seeded with the
K strongest (meaning, symbol) entries of one randomly chosen veteran, at half
weight. K is the transmission budget: K = 0 means no inheritance, K = 12 is
roughly one entry per meaning.

Run: python3 seeded_turnover.py --seed 1 --trials 20 --out results_seeded
"""
import argparse
import json
import os
import random

from conventions import (Agent, interact, consensus, ambiguity, M, POP, EPOCH,
                         EPOCHS, WARMUP, CONSONANTS, VOWELS)

SYMBOLS = 12


def seed_from(donor, newcomer, k):
    entries = sorted(((w, m, s) for m, d in donor.lex.items() for s, w in d.items()),
                     reverse=True)[:k]
    for w, m, s in entries:
        newcomer.add(m, s, w / 2)


def run(turn, k, seed):
    rng = random.Random(seed)
    allsyms = [c + v for c in CONSONANTS for v in VOWELS]
    pool = rng.sample(allsyms, SYMBOLS)
    pop = [Agent(0) for _ in range(POP)]
    prev, rows = None, []
    for e in range(EPOCHS):
        recent = []
        for _ in range(EPOCH):
            ok, _ = interact(pop, rng, pool)
            recent.append(ok)
        cons = consensus(pop)
        ret = None
        if prev is not None:
            ret = sum(1 for m in range(M)
                      if prev[m] is not None and cons[m] == prev[m]) / M
        rows.append((sum(recent[-100:]) / 100, ambiguity(cons), ret,
                     sum(recent[:50]) / 50))
        prev = cons
        if e < EPOCHS - 1:
            veterans = list(pop)
            for idx in rng.sample(range(POP), round(turn * POP)):
                newcomer = Agent(0)
                if k > 0:
                    seed_from(rng.choice(veterans), newcomer, k)
                pop[idx] = newcomer
    late = rows[WARMUP:]
    n = len(late)
    return {
        'success_end': sum(r[0] for r in late) / n,
        'success_start': sum(r[3] for r in late) / n,
        'ambiguity': sum(r[1] for r in late) / n,
        'retention': sum(r[2] for r in late) / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--trials', type=int, default=20)
    ap.add_argument('--out', default='results_seeded')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    summary = {}
    for turn in (0.5, 0.8, 1.0):
        for k in (0, 3, 6, 12):
            acc = {}
            for t in range(args.trials):
                res = run(turn, k, args.seed * 100000 + int(turn * 100) * 100 + t)
                for key, v in res.items():
                    acc[key] = acc.get(key, 0.0) + v / args.trials
            summary['turnover=%.1f,K=%d' % (turn, k)] = {key: round(v, 3) for key, v in acc.items()}
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    for key, v in summary.items():
        print(key, v)


if __name__ == '__main__':
    main()
