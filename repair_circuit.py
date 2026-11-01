"""Budgeted repair of corrupted boolean circuits.

A circuit has NIN inputs and NG gates. Gate i is (type, a, b) where a and b
index earlier signals (inputs first, then earlier gates). The output is the
last gate. Damage is k random point edits (change a gate type or rewire one
input). A repair step is also a point edit with cost 1. The repairer sees only
VISIBLE of the 2**NIN truth-table rows; afterwards the repaired circuit is
checked on all rows, so the run reports both visible success and true
correctness. Memory is the beam width; disk output is capped by CappedWriter.

Run: python3 repair_circuit.py --seed 1 --trials 100 --out results_circuit
"""
import argparse
import json
import os
import random

from repair import CappedWriter

NIN = 4
NG = 6
VISIBLE = 10
TYPES = ('and', 'or', 'xor', 'nand')
ROWS = list(range(2 ** NIN))


def gate(t, a, b):
    if t == 'and':
        return a & b
    if t == 'or':
        return a | b
    if t == 'xor':
        return a ^ b
    return 1 - (a & b)


def run(circ, row):
    sig = [(row >> i) & 1 for i in range(NIN)]
    for t, a, b in circ:
        sig.append(gate(t, sig[a], sig[b]))
    return sig[-1]


def random_circuit(rng):
    return tuple((rng.choice(TYPES), rng.randrange(NIN + i), rng.randrange(NIN + i))
                 for i in range(NG))


def edits(circ):
    for i, (t, a, b) in enumerate(circ):
        for t2 in TYPES:
            if t2 != t:
                yield circ[:i] + ((t2, a, b),) + circ[i + 1:]
        for a2 in range(NIN + i):
            if a2 != a:
                yield circ[:i] + ((t, a2, b),) + circ[i + 1:]
        for b2 in range(NIN + i):
            if b2 != b:
                yield circ[:i] + ((t, a, b2),) + circ[i + 1:]


def corrupt(circ, rng, k):
    for _ in range(k):
        circ = rng.choice(list(edits(circ)))
    return circ


def error(circ, spec):
    return sum(run(circ, r) != y for r, y in spec)


def repair(broken, spec, memory, max_edits):
    beam = [broken]
    seen = {broken}
    examined = 0
    for depth in range(max_edits + 1):
        for c in beam:
            if error(c, spec) == 0:
                return c, depth, examined
        pool = []
        for c in beam:
            for n in edits(c):
                if n not in seen:
                    seen.add(n)
                    pool.append(n)
                    examined += 1
        if not pool:
            break
        pool.sort(key=lambda c: error(c, spec))
        beam = pool[:memory]
        if len(seen) > 50 * memory:
            seen = set(beam)
    return None, None, examined


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--trials', type=int, default=100)
    ap.add_argument('--out', default='results_circuit')
    ap.add_argument('--ceiling', type=int, default=20000)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    detail = CappedWriter(os.path.join(args.out, 'detail.jsonl'), args.ceiling)
    summary = {}
    for k in (1, 2, 3):
        for memory in (2, 8, 32, 128):
            vis = true_ok = within = ex_total = 0
            r = random.Random(args.seed * 1000 + k * 17 + memory)
            for t in range(args.trials):
                truth = random_circuit(r)
                shown = r.sample(ROWS, VISIBLE)
                spec = [(row, run(truth, row)) for row in shown]
                broken = corrupt(truth, r, k)
                fixed, used, ex = repair(broken, spec, memory, k + 2)
                ex_total += ex
                if fixed is not None:
                    vis += 1
                    if all(run(fixed, row) == run(truth, row) for row in ROWS):
                        true_ok += 1
                    if used <= k:
                        within += 1
                if t < 2:
                    detail.write({'k': k, 'memory': memory, 'trial': t,
                                  'truth': truth, 'broken': broken,
                                  'fixed': fixed, 'edits': used, 'examined': ex})
            n = args.trials
            summary['k=%d,memory=%d' % (k, memory)] = {
                'visible_ok': vis / n,
                'correct_all_trials': true_ok / n,
                'correct_given_visible': (true_ok / vis) if vis else None,
                'within_k_edits': within / n,
                'mean_examined': ex_total / n,
            }
    detail.close()
    summary['detail_bytes'] = detail.used
    summary['detail_lines_dropped'] = detail.dropped
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    for key, v in summary.items():
        print(key, v)


if __name__ == '__main__':
    main()
