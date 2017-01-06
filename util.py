import sys, csv
from itertools import product


def progressbar(it, prefix="", size=60):
    count = len(it)

    def _show(_i):
        x = int(size * _i / count)
        sys.stdout.write("%s[%s%s] %i/%i\r" % (prefix, "#" * x,
                                               "." * (size - x), _i, count))
        sys.stdout.flush()

    _show(0)
    for i, item in enumerate(it):
        yield item
        _show(i + 1)
    sys.stdout.write("\n")
    sys.stdout.flush()


def exp_builder(num, *params):
    perm = product(*params)
    exp = []
    for i, p in enumerate(perm):
        exp.append([i + num] + list(p))

    return exp


def exp_writer(name, exp):
    with open(name, 'wb') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(exp)
