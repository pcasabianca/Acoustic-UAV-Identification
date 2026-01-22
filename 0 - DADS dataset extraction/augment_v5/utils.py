import hashlib


def stable_int_seed(s: str) -> int:
    h = hashlib.sha256()
    h.update(s.encode())
    # return a 32-bit integer
    return int(h.hexdigest()[:8], 16)


def distribute_counts(n_total, ratios):
    floats = [n_total * r for r in ratios]
    floors = [int(x) for x in floats]
    rem = n_total - sum(floors)
    fracs = [(i, floats[i] - floors[i]) for i in range(len(floors))]
    fracs.sort(key=lambda x: x[1], reverse=True)
    for i in range(rem):
        idx = fracs[i][0]
        floors[idx] += 1
    return floors
