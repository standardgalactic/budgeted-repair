"""Convention formation and repair in small agent populations.

Agents are ordinary Python objects holding a weighted lexicon that maps
meanings to symbols. Pairs of agents play a naming game: a speaker names a
meaning, a hearer guesses the meaning, and a failure is repaired by the
speaker revealing the intended meaning. Three budgets shape the outcome.

Symbol inventory S: the number of distinct symbols the community can produce.
When S is smaller than the number of meanings, compression forces ambiguity.
Lexicon capacity CAP: each agent forgets its weakest entries beyond this size.
Correction history H: each agent keeps its last H corrections (symbol and
intended meaning) and consults them when interpreting a symbol. H = 0 disables
the history.

Every EPOCH interactions a fraction of the population is replaced by naive
newcomers, which tests whether a convention survives turnover.

Run: python3 conventions.py --seed 1 --trials 20 --out results_language
"""
import argparse
import json
import os
import random
from collections import Counter, deque

M = 12          # meanings
POP = 10        # agents
EPOCH = 400     # interactions between turnover events
EPOCHS = 15
TURN = 0.2      # fraction replaced at each turnover
CAP = 24        # lexicon entries per agent
BONUS = 0.3     # weight a logged correction adds when interpreting
WARMUP = 3      # epochs ignored when averaging
CONSONANTS = 'ptkmnslr'
VOWELS = 'aiu'


class Agent:
    def __init__(self, history):
        self.lex = {}
        self.log = deque(maxlen=history) if history > 0 else None

    def size(self):
        return sum(len(d) for d in self.lex.values())

    def add(self, m, s, dw):
        d = self.lex.setdefault(m, {})
        d[s] = min(1.0, d.get(s, 0.0) + dw)
        if d[s] <= 1e-9:
            del d[s]
        if not d:
            del self.lex[m]
        while self.size() > CAP:
            w, m2, s2 = min((w, m2, s2) for m2, d2 in self.lex.items()
                            for s2, w in d2.items())
            del self.lex[m2][s2]
            if not self.lex[m2]:
                del self.lex[m2]

    def speak(self, m, rng, pool):
        d = self.lex.get(m)
        if not d:
            s = rng.choice(pool)
            self.add(m, s, 0.5)
            return s
        best = max(d.values())
        return rng.choice([s for s, w in d.items() if w == best])

    def hear(self, s, rng):
        logged = Counter(mm for ss, mm in self.log if ss == s) if self.log else {}
        best, cands = 0.0, []
        for m in range(M):
            sc = self.lex.get(m, {}).get(s, 0.0) + BONUS * logged.get(m, 0)
            if sc > best + 1e-9:
                best, cands = sc, [m]
            elif sc > 0 and abs(sc - best) <= 1e-9:
                cands.append(m)
        return rng.choice(cands) if cands else rng.randrange(M)


def interact(pop, rng, pool):
    sp, hr = rng.sample(pop, 2)
    m = rng.randrange(M)
    s = sp.speak(m, rng, pool)
    g = hr.hear(s, rng)
    if g == m:
        for ag in (sp, hr):
            ag.add(m, s, 0.1)
            for s2 in list(ag.lex.get(m, {})):
                if s2 != s:
                    ag.add(m, s2, -0.05)
        return True, (m, s, g)
    hr.add(m, s, 0.3)
    if g in hr.lex and s in hr.lex[g]:
        hr.add(g, s, -0.1)
    if hr.log is not None:
        hr.log.append((s, m))
    sp.add(m, s, -0.1)
    return False, (m, s, g)


def consensus(pop):
    cons = {}
    for m in range(M):
        tops = Counter()
        for ag in pop:
            d = ag.lex.get(m)
            if d:
                best = max(d.values())
                tops[min(s for s, w in d.items() if w == best)] += 1
        cons[m] = tops.most_common(1)[0][0] if tops else None
    return cons


def ambiguity(cons):
    syms = [s for s in cons.values() if s is not None]
    count = Counter(syms)
    return sum(1 for s in syms if count[s] > 1) / M


class Transcript:
    def __init__(self, path, ceiling):
        self.f = open(path, 'w')
        self.ceiling, self.used, self.dropped = ceiling, 0, 0

    def line(self, text):
        if self.used + len(text) + 1 > self.ceiling:
            self.dropped += 1
            return
        self.f.write(text + '\n')
        self.used += len(text) + 1

    def close(self):
        self.f.close()


def run(S, H, seed, transcript=None):
    rng = random.Random(seed)
    allsyms = [c + v for c in CONSONANTS for v in VOWELS]
    pool = rng.sample(allsyms, S)
    pop = [Agent(H) for _ in range(POP)]
    prev, rows = None, []
    for e in range(EPOCHS):
        recent = []
        for i in range(EPOCH):
            ok, (m, s, g) = interact(pop, rng, pool)
            recent.append(ok)
            if transcript and e == 0 and i < 10:
                transcript.line('epoch 0 turn %d: meaning %d named %s, heard %d, %s'
                                % (i, m, s, g, 'ok' if ok else 'repair'))
        cons = consensus(pop)
        succ = sum(recent[-100:]) / 100
        amb = ambiguity(cons)
        ret = None
        if prev is not None:
            kept = sum(1 for m in range(M) if prev[m] is not None and cons[m] == prev[m])
            ret = kept / M
        rows.append((succ, amb, ret))
        if transcript:
            transcript.line('epoch %d: success %.2f ambiguity %.2f retention %s'
                            % (e, succ, amb, 'n/a' if ret is None else '%.2f' % ret))
        prev = cons
        if e < EPOCHS - 1:
            for idx in rng.sample(range(POP), round(TURN * POP)):
                pop[idx] = Agent(H)
            if transcript:
                transcript.line('turnover: %d agents replaced' % round(TURN * POP))
    late = rows[WARMUP:]
    return {
        'success': sum(r[0] for r in late) / len(late),
        'ambiguity': sum(r[1] for r in late) / len(late),
        'retention': sum(r[2] for r in late) / len(late),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--trials', type=int, default=20)
    ap.add_argument('--out', default='results_language')
    ap.add_argument('--ceiling', type=int, default=6000)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tr = Transcript(os.path.join(args.out, 'transcript.txt'), args.ceiling)
    summary = {}
    for S in (6, 12, 24):
        for H in (0, 8, 32):
            acc = {'success': 0.0, 'ambiguity': 0.0, 'retention': 0.0}
            for t in range(args.trials):
                keep = tr if (S == 12 and H == 8 and t == 0) else None
                if keep:
                    tr.line('--- selected run: symbols %d, history %d ---' % (S, H))
                res = run(S, H, args.seed * 100000 + S * 1000 + H * 10 + t, keep)
                for k in acc:
                    acc[k] += res[k] / args.trials
            summary['symbols=%d,history=%d' % (S, H)] = {k: round(v, 3) for k, v in acc.items()}
    tr.close()
    summary['transcript_bytes'] = tr.used
    summary['transcript_lines_dropped'] = tr.dropped
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    for key, v in summary.items():
        print(key, v)


if __name__ == '__main__':
    main()
