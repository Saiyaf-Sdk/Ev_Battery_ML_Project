"""Create sample API request from the first CSV row (demonstration only)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd
from .config import IDENTIFIERS, TARGET, SUSPECT_SCORES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True, type=Path)
    p.add_argument('--out', type=Path, default=Path('artifacts/example_request.json'))
    args = p.parse_args()
    row = pd.read_csv(args.data, nrows=1).iloc[0]
    exclude = {TARGET, *IDENTIFIERS, *SUSPECT_SCORES}
    values = {k: (None if pd.isna(v) else v.item() if hasattr(v, 'item') else v)
              for k, v in row.items() if k not in exclude}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'features': values}, indent=2), encoding='utf-8')
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
