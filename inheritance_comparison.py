"""Controlled comparison of what newcomers inherit under full turnover.

At each cycle the entire population is replaced. Every newcomer receives up to
K (meaning, symbol) pairs from one randomly chosen donor, always inserted into
its lexicon at the same fixed weight, so the only difference between regimes is
which K pairs are chosen:

  strong       the donor's K highest-weight lexicon entries
  corrections  the donor's K most recent distinct logged corrections
  random       K lexicon entries drawn at random from the donor
  none         nothing (K is ignored)

All agents keep the same correction-log capacity in every regime, so the log is
available for interpretation everywhere and only the inheritance differs.

Measured per cycle: how many pairs were actually delivered, what fraction of
them agree with the community consensus held before replacement, retention of
that consensus immediately after replacement (before any interaction), after
100 interactions, and after 400, plus success in the first and last 100
interactions. Each run averages over its cycles; the summary reports the mean
and standard error across independent runs.

Run: python3 inheritance_comparison.py --seed 1 --trials 30 --out results_inheritance
"""
import argparse
import json
import math
import os
import random

from conventions import (Agent, interact, consensus, M, POP, EPOCH,
                         CONSONANTS, VOWELS)

SYMBOLS = 12
HISTORY = 32
CYCLES = 10
WARM = 3 * EPOCH
INHERIT_WEIGHT = 0.25


def retention(old, new):
    return sum(1 for m in range(M) if old[m] is not None and new[m] == old[m]) / M


def pick(donor, mode, k, rng):
    if mode == 'strong':
        ent = sorted(((w, m, s) for m, d in donor.lex.items() for s, w in d.items()),
                     reverse=True)[:k]
        return [(m, s) for w, m, s in ent]
    if mode == 'strong_random_ties':
        ent = sorted(((w, rng.random(), m, s) for m, d in donor.lex.items()
                      for s, w in d.items()), reverse=True)[:k]
        return [(m, s) for w, r, m, s in ent]
    if mode == 'random':
        ent = [(m, s) for m, d in donor.lex.items() for s in d]
        return rng.sample(ent, min(k, len(ent)))
    out, seen = [], set()
    for s, m in reversed(donor.log):
        if (m, s) not in seen:
            seen.add((m, s))
            out.append((m, s))
        if len(out) == k:
            break
    return out


def run(mode, k, seed):
    rng = random.Random(seed)
    allsyms = [c + v for c in CONSONANTS for v in VOWELS]
    pool = rng.sample(allsyms, SYMBOLS)
    pop = [Agent(HISTORY) for _ in range(POP)]
    for _ in range(WARM):
        interact(pop, rng, pool)
    acc = {key: 0.0 for key in ('delivered', 'agree', 'ret0', 'ret100', 'ret400',
                                'succ_first100', 'succ_last100')}
    for _ in range(CYCLES):
        old = consensus(pop)
        donors = list(pop)
        delivered = agreeing = 0
        new = [Agent(HISTORY) for _ in range(POP)]
        if mode != 'none':
            for ag in new:
                for m, s in pick(rng.choice(donors), mode, k, rng):
                    ag.add(m, s, INHERIT_WEIGHT)
                    delivered += 1
                    agreeing += (old[m] == s)
        pop = new
        acc['delivered'] += delivered / POP
        acc['agree'] += (agreeing / delivered) if delivered else 0.0
        acc['ret0'] += retention(old, consensus(pop))
        first, last = [], []
        for i in range(EPOCH):
            ok, _ = interact(pop, rng, pool)
            if i < 100:
                first.append(ok)
            if i >= EPOCH - 100:
                last.append(ok)
            if i == 99:
                acc['ret100'] += retention(old, consensus(pop))
        acc['ret400'] += retention(old, consensus(pop))
        acc['succ_first100'] += sum(first) / 100
        acc['succ_last100'] += sum(last) / 100
    return {key: v / CYCLES for key, v in acc.items()}


def mean_sem(xs):
    n = len(xs)
    mu = sum(xs) / n
    var = sum((x - mu) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
    return round(mu, 3), round(math.sqrt(var / n), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--trials', type=int, default=30)
    ap.add_argument('--out', default='results_inheritance')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    conds = [('none', 0)] + [(m, k) for k in (3, 6, 12)
                             for m in ('strong', 'corrections', 'random')]
    summary = {}
    for mode, k in conds:
        runs = [run(mode, k, args.seed * 100000 + 7 * t) for t in range(args.trials)]
        summary['%s,K=%d' % (mode, k)] = {
            key: mean_sem([r[key] for r in runs]) for key in runs[0]}
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print('values are (mean, standard error) across %d independent runs' % args.trials)
    for key, v in summary.items():
        print(key, v)


if __name__ == '__main__':
    main()
