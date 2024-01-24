from __future__ import annotations

from json import JSONEncoder, dump
from typing import Any, Iterator

import pandas

from collections import defaultdict


class Serialize(JSONEncoder):

    def iterencode(self, o: Any, _one_shot: bool = ...) -> Iterator[str]:
        if isinstance(o, defaultdict):
            for k, v in o.items():
                if isinstance(v, defaultdict):
                    o[k] = {label: count for label, count in sorted(v.items(), key=lambda x: x[1], reverse=True)}
        return super().iterencode(o, _one_shot)


def nested_dict(n: int, t: type):
    if n == 1:
        return defaultdict(t)
    else:
        return defaultdict(lambda: nested_dict(n - 1, t))


def clean(string: str):
    if not isinstance(string, str):
        return ""
    return string.replace('[', '').replace(']', '').replace('\\', '').replace("'", '')


df = pandas.read_csv('./data/out.csv', delimiter=',', usecols=['msc', 'keyword', 'refs'])


def generate_idx(idx_name):
    km_idx = nested_dict(2, int)
    mk_idx = nested_dict(2, int)
    latest_progress = 0
    for row in df.itertuples():
        current_progress = round(row[0] / len(df) * 100, 1)
        if current_progress != latest_progress:
            print(current_progress, '%')
            latest_progress = current_progress
        mscs = clean(row.msc).replace(' ', '').split(',')
        for keyword in clean(getattr(row, idx_name)).split(","):
            keyword = keyword.lstrip().rstrip()
            for clea_str in [',', "'", '"', "`", '\\']:
                keyword = keyword.strip(clea_str)
            if '' == keyword:
                continue
            for msc in mscs:
                km_idx[msc][keyword] += 1
                mk_idx[keyword][msc] += 1

    with open(f"{idx_name}_msc_idx.json", 'w') as f:
        dump(km_idx, f, cls=Serialize)

    with open(f"msc_{idx_name}_idx.json", 'w') as f:
        dump(mk_idx, f, cls=Serialize)


print('generate keyword index')
generate_idx('keyword')
print('generate reference index')
generate_idx('refs')
