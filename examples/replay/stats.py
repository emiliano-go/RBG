def median(xs):
    ordered = sorted(xs)
    n = len(ordered)
    return ordered[n // 2]
