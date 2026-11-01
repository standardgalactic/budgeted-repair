"""Budgeted repair of corrupted dependency graphs.

A graph on N nodes is a frozenset of directed edges (u, v), meaning u must be
built before v. The specification is a set of reachability facts for a random
subset of ordered pairs (is u required before v, yes or no), plus the
requirement that the graph is acyclic. Damage is a number of random edge
toggles (add a missing edge or delete a present one). A repair step is also an
edge toggle with cost 1. As in repair.py, working memory is the beam width and
disk output is capped by CappedWriter.

Run: python3 repair_graph.py --seed 1 --trials 100 --out results_graph
"""
import argparse
import json
import os
import random

from repair import CappedWriter

N = 8
EDGE_P = 0.25
TESTED_FRACTION = 0.6
CYCLE_PENALTY = 5


def closure(edges):
    adj = {i: [] for i in range(N)}
    for u, v in edges:
        adj[u].append(v)
    reach = set()
    for s in range(N):
        stack = list(adj[s])
        seen = set()
        while stack:
            t = stack.pop()
            if t in seen:
                continue
            seen.add(t)
            stack.extend(adj[t])
        for t in seen:
            reach.add((s, t))
    return reach


def error(edges, spec):
    reach = closure(edges)
    cyc = any((i, i) in reach for i in range(N))
    mism = sum((p in reach) != val for p, val in spec)
    return mism + (CYCLE_PENALTY if cyc else 0)


def all_pairs():
    return [(u, v) for u in range(N) for v in range(N) if u != v]


def toggles(edges):
    for p in all_pairs():
        yield edges - {p} if p in edges else edges | {p}


def random_dag(rng):
    order = list(range(N))
    rng.shuffle(order)
    rank = {n: i for i, n in enumerate(order)}
    return frozenset((u, v) for u, v in all_pairs()
                     if rank[u] < rank[v] and rng.random() < EDGE_P)


def corrupt(edges, rng, k):
    for _ in range(k):
        edges = rng.choice(list(toggles(edges)))
    return edges


def repair(broken, spec, memory, max_edits):
    beam = [broken]
    seen = {broken}
    examined = 0
    for depth in range(max_edits + 1):
        for g in beam:
            if error(g, spec) == 0:
                return depth, examined
        pool = []
        for g in beam:
            for n in toggles(g):
                if n not in seen:
                    seen.add(n)
                    pool.append(n)
                    examined += 1
        if not pool:
            break
        pool.sort(key=lambda g: error(g, spec))
        beam = pool[:memory]
        if len(seen) > 50 * memory:
            seen = set(beam)
    return None, examined


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--trials', type=int, default=100)
    ap.add_argument('--out', default='results_graph')
    ap.add_argument('--ceiling', type=int, default=20000)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    detail = CappedWriter(os.path.join(args.out, 'detail.jsonl'), args.ceiling)
    summary = {}
    for k in (2, 4, 6):
        for memory in (2, 8, 32, 128):
            solved = within = examined_total = 0
            r = random.Random(args.seed * 1000 + k * 17 + memory)
            for t in range(args.trials):
                truth = random_dag(r)
                reach = closure(truth)
                tested = r.sample(all_pairs(), int(TESTED_FRACTION * N * (N - 1)))
                spec = [(p, p in reach) for p in tested]
                broken = corrupt(truth, r, k)
                used, ex = repair(broken, spec, memory, max_edits=k + 2)
                examined_total += ex
                if used is not None:
                    solved += 1
                    if used <= k:
                        within += 1
                if t < 2:
                    detail.write({'k': k, 'memory': memory, 'trial': t,
                                  'truth': sorted(truth), 'broken': sorted(broken),
                                  'edits': used, 'examined': ex})
            summary['k=%d,memory=%d' % (k, memory)] = {
                'solved': solved / args.trials,
                'within_k_edits': within / args.trials,
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
