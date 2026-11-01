"""Matched-state comparison of inheritance rules under full replacement.

One reference community is evolved without any replacement. Every ECYCLE
interactions its complete state is taken as a snapshot, and every inheritance
rule and budget K is applied to that same snapshot using the same donor
assignment (newcomer i copies from donor assign[i]). Each branch then runs
EPOCH interactions with the same random seed. Because every branch starts from
identical donors, differences between rules measure the immediate effect of
content selection and not accumulated effects of earlier policies on the donor
population. The unmatched, cumulative design is in inheritance_comparison.py.

Definitions
  consensus(pop): for each meaning, each agent with any entry votes for its
    highest-weight symbol (ties inside an agent go to the alphabetically
    smallest symbol); the most common vote wins (ties across agents go to the
    symbol voted first in agent order). A meaning with no entries in any agent
    has consensus None.
  retention(old, new): the number of meanings m with old[m] not None and
    new[m] == old[m], divided by the 12 meanings. A meaning with no inherited
    entry has consensus None and counts as not retained, so a missing
    convention never counts as a retained one. Chance agreement for an
    unrelated convention is about 1/S for S symbols when every meaning has an
    old consensus.
  strong breaks weight ties by the largest meaning index, so newcomers
    share the same meanings; strong_random_ties breaks them at random.
    pop_coverage: share of the 12 meanings carried by at least one newcomer.
  delivered: pairs actually inserted per newcomer (fewer than K if the donor
    has fewer eligible items). distinct: distinct meanings covered per
    newcomer. conflicts: meanings for which one newcomer received more than
    one symbol. agree: share of delivered pairs equal to the old consensus.
  Repeated pairs: the corrections rule uses distinct (meaning, symbol) pairs
    only, so a pair logged several times is delivered once. Random and strong
    draw from distinct lexicon entries, which are unique by construction.

Each independent run uses its own reference community. Cycles within a run are
averaged and are not treated as independent. Reported uncertainty is the
standard error across runs; differences between rules are paired by run and by
snapshot.

Run: python3 matched_inheritance.py --seed 1 --trials 30 --out results_matched
"""
import argparse
import json
import math
import os
import random

from conventions import (Agent, interact, consensus, M, POP, EPOCH,
                         CONSONANTS, VOWELS)
from inheritance_comparison import (pick, retention, SYMBOLS, HISTORY, CYCLES,
                                    WARM, INHERIT_WEIGHT)

BUDGETS = (3, 6, 12)
MODES = ('strong', 'strong_random_ties', 'corrections', 'random')
KEYS = ('delivered', 'distinct', 'pop_coverage', 'conflicts', 'agree', 'ret0',
        'ret100', 'ret400', 'succ_first100', 'succ_last100')


def branch(donors, assign, old, mode, k, pool, seed):
    sel = random.Random(seed + 1)
    rng = random.Random(seed)
    new = [Agent(HISTORY) for _ in range(POP)]
    deliv = dist = conf = agree = 0
    union = set()
    if mode != 'none':
        for i, ag in enumerate(new):
            pairs = pick(donors[assign[i]], mode, k, sel)
            syms_by_m = {}
            for m, s in pairs:
                ag.add(m, s, INHERIT_WEIGHT)
                syms_by_m.setdefault(m, set()).add(s)
                union.add(m)
                agree += (old[m] == s)
            deliv += len(pairs)
            dist += len(syms_by_m)
            conf += sum(1 for v in syms_by_m.values() if len(v) > 1)
    out = {'delivered': deliv / POP, 'distinct': dist / POP,
           'pop_coverage': len(union) / M, 'conflicts': conf / POP,
           'agree': (agree / deliv) if deliv else 0.0,
           'ret0': retention(old, consensus(new))}
    first, last = [], []
    for i in range(EPOCH):
        ok, _ = interact(new, rng, pool)
        if i < 100:
            first.append(ok)
        if i >= EPOCH - 100:
            last.append(ok)
        if i == 99:
            out['ret100'] = retention(old, consensus(new))
    out['ret400'] = retention(old, consensus(new))
    out['succ_first100'] = sum(first) / 100
    out['succ_last100'] = sum(last) / 100
    return out


def run(seed):
    rng = random.Random(seed)
    allsyms = [c + v for c in CONSONANTS for v in VOWELS]
    pool = rng.sample(allsyms, SYMBOLS)
    ref = [Agent(HISTORY) for _ in range(POP)]
    for _ in range(WARM):
        interact(ref, rng, pool)
    conds = [('none', 0)] + [(m, k) for k in BUDGETS for m in MODES]
    acc = {c: {key: 0.0 for key in KEYS} for c in conds}
    extra = {'coverage_old': 0.0, 'ref_stability': 0.0}
    arng = random.Random(seed + 5)
    for cyc in range(CYCLES):
        old = consensus(ref)
        extra['coverage_old'] += sum(1 for m in range(M) if old[m] is not None) / M
        assign = [arng.randrange(POP) for _ in range(POP)]
        donors = list(ref)
        bseed = seed * 31 + cyc * 1000
        for mode, k in conds:
            res = branch(donors, assign, old, mode, k, pool, bseed)
            for key in KEYS:
                acc[(mode, k)][key] += res[key] / CYCLES
        for _ in range(EPOCH):
            interact(ref, rng, pool)
        extra['ref_stability'] += retention(old, consensus(ref)) / CYCLES
    extra['coverage_old'] /= CYCLES
    return acc, extra


def mean_sem(xs):
    n = len(xs)
    mu = sum(xs) / n
    var = sum((x - mu) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
    return [round(mu, 3), round(math.sqrt(var / n), 3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--trials', type=int, default=30)
    ap.add_argument('--out', default='results_matched')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    runs, extras = [], []
    for t in range(args.trials):
        a, e = run(args.seed * 100000 + 7 * t)
        runs.append(a)
        extras.append(e)
    summary = {'n_runs': args.trials,
               'reference': {k: mean_sem([e[k] for e in extras]) for k in extras[0]},
               'rules': {}, 'paired_differences': {}}
    for cond in runs[0]:
        name = '%s,K=%d' % cond
        summary['rules'][name] = {key: mean_sem([r[cond][key] for r in runs])
                                  for key in KEYS}
    for k in BUDGETS:
        for a, b in (('strong', 'corrections'), ('strong', 'random'),
                     ('corrections', 'random'), ('strong_random_ties', 'random'),
                     ('strong_random_ties', 'corrections'),
                     ('strong', 'strong_random_ties')):
            name = '%s minus %s,K=%d' % (a, b, k)
            summary['paired_differences'][name] = {
                key: mean_sem([r[(a, k)][key] - r[(b, k)][key] for r in runs])
                for key in ('ret0', 'ret400', 'succ_last100', 'agree', 'distinct',
                            'pop_coverage')}
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print('values are [mean, standard error] across %d runs' % args.trials)
    print('reference', summary['reference'])
    for name, v in summary['rules'].items():
        print(name, v)
    for name, v in summary['paired_differences'].items():
        print(name, v)


if __name__ == '__main__':
    main()
