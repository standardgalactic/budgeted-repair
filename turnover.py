"""Convention survival under heavy turnover, with and without inherited logs.

Reuses the agents and naming game from conventions.py. At each turnover a
fraction of the population is replaced by naive newcomers. Three regimes are
compared: no correction history, a private history that newcomers start
without, and a history that each newcomer inherits as a copy of a randomly
chosen departing or veteran agent's log. A turnover fraction of 1.0 replaces
the whole population, so a convention can only survive through what is
transmitted in the inherited logs.

Run: python3 turnover.py --seed 1 --trials 20 --out results_turnover
"""
import argparse
import json
import os
import random
from collections import deque

from conventions import (Agent, interact, consensus, ambiguity, M, POP, EPOCH,
                         EPOCHS, WARMUP, CONSONANTS, VOWELS)

SYMBOLS = 12
HISTORY = 32


def run(turn, history, inherit, seed):
    rng = random.Random(seed)
    allsyms = [c + v for c in CONSONANTS for v in VOWELS]
    pool = rng.sample(allsyms, SYMBOLS)
    pop = [Agent(history) for _ in range(POP)]
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
                newcomer = Agent(history)
                if inherit and history > 0:
                    donor = rng.choice(veterans)
                    newcomer.log = deque(donor.log, maxlen=history)
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
    ap.add_argument('--out', default='results_turnover')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    regimes = (('no_history', 0, False),
               ('own_history', HISTORY, False),
               ('inherited_history', HISTORY, True))
    summary = {}
    for turn in (0.5, 0.8, 1.0):
        for name, h, inh in regimes:
            acc = {}
            for t in range(args.trials):
                res = run(turn, h, inh, args.seed * 100000 + int(turn * 100) * 100 + t)
                for k, v in res.items():
                    acc[k] = acc.get(k, 0.0) + v / args.trials
            summary['turnover=%.1f,%s' % (turn, name)] = {k: round(v, 3) for k, v in acc.items()}
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    for key, v in summary.items():
        print(key, v)


if __name__ == '__main__':
    main()
