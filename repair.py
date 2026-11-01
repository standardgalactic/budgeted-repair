"""Budgeted repair of corrupted arithmetic expressions.

An expression is a nested tuple (op, left, right), the string 'x', or an int.
A corrupted copy is produced by random point edits (operator change, constant
change, leaf swap). A repairer must restore behaviour on a fixed test set
using single-point edits, each with an explicit cost, while holding at most
`memory` candidate states at once. Output is capped by a byte ceiling that is
enforced by the writer itself.

Run: python3 repair.py --seed 1 --trials 200 --out results_expression
"""
import argparse
import json
import os
import random

OPS = ('+', '-', '*')
CONSTS = (0, 1, 2, 3)
TESTS = tuple(range(-3, 5))


def evaluate(e, x):
    if e == 'x':
        return x
    if isinstance(e, int):
        return e
    op, a, b = e
    l, r = evaluate(a, x), evaluate(b, x)
    return l + r if op == '+' else l - r if op == '-' else l * r


def random_expr(rng, depth):
    if depth == 0 or rng.random() < 0.25:
        return 'x' if rng.random() < 0.5 else rng.choice(CONSTS)
    return (rng.choice(OPS), random_expr(rng, depth - 1), random_expr(rng, depth - 1))


def single_edits(e):
    """Yield every expression one point edit away from e."""
    if e == 'x' or isinstance(e, int):
        for c in CONSTS:
            if c != e:
                yield c
        if e != 'x':
            yield 'x'
        else:
            return
        return
    op, a, b = e
    for o in OPS:
        if o != op:
            yield (o, a, b)
    for na in single_edits(a):
        yield (op, na, b)
    for nb in single_edits(b):
        yield (op, a, nb)


def error(e, spec):
    return sum(abs(evaluate(e, x) - y) for x, y in spec)


def corrupt(e, rng, k):
    for _ in range(k):
        edits = list(single_edits(e))
        e = rng.choice(edits)
    return e


def repair(broken, spec, memory, max_edits):
    """Beam search. `memory` bounds the number of states kept per round.
    Returns (edits_used, states_examined) or (None, states_examined)."""
    beam = [broken]
    seen = {broken}
    examined = 0
    for depth in range(max_edits + 1):
        for e in beam:
            if error(e, spec) == 0:
                return depth, examined
        pool = []
        for e in beam:
            for n in single_edits(e):
                if n not in seen:
                    seen.add(n)
                    pool.append(n)
                    examined += 1
        if not pool:
            break
        pool.sort(key=lambda n: error(n, spec))
        beam = pool[:memory]
        if len(seen) > 50 * memory:
            seen = set(beam)  # forget history under pressure
    return None, examined


class CappedWriter:
    """Refuses to exceed a byte ceiling; detail lines degrade to counts."""

    def __init__(self, path, ceiling):
        self.f = open(path, 'w')
        self.ceiling = ceiling
        self.used = 0
        self.dropped = 0

    def write(self, obj):
        line = json.dumps(obj, separators=(',', ':')) + '\n'
        if self.used + len(line) > self.ceiling:
            self.dropped += 1
            return False
        self.f.write(line)
        self.used += len(line)
        return True

    def close(self):
        self.f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--trials', type=int, default=200)
    ap.add_argument('--out', default='results_expression')
    ap.add_argument('--ceiling', type=int, default=20000)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    detail = CappedWriter(os.path.join(args.out, 'detail.jsonl'), args.ceiling)
    summary = {}
    for k in (1, 2, 3):
        for memory in (2, 8, 32, 128):
            solved = optimal = 0
            examined_total = 0
            r = random.Random(args.seed * 1000 + k * 17 + memory)
            for t in range(args.trials):
                truth = random_expr(r, 3)
                spec = [(x, evaluate(truth, x)) for x in TESTS]
                broken = corrupt(truth, r, k)
                used, ex = repair(broken, spec, memory, max_edits=k + 2)
                examined_total += ex
                if used is not None:
                    solved += 1
                    if used <= k:
                        optimal += 1
                if t < 2:  # keep detailed histories only for selected cases
                    detail.write({'k': k, 'memory': memory, 'trial': t,
                                  'truth': truth, 'broken': broken,
                                  'edits': used, 'examined': ex})
            summary['k=%d,memory=%d' % (k, memory)] = {
                'solved': solved / args.trials,
                'within_k_edits': optimal / args.trials,
                'mean_examined': examined_total / args.trials,
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
