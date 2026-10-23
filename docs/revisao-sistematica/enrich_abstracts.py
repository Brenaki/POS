"""Complete truncated/missing abstracts of unicos.csv from OpenAlex.

Usage: python prisma_merge.py && python enrich_abstracts.py && python prisma_merge.py
Looks each record up by DOI, else by exact-title search, and writes
raw_enrich/abstracts.json ({uid: {doi, abstract, openalex_id}}). The uid is
stable across re-merges because the raw inputs and their order do not change.
"""

import csv
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

from openalex_search import _abstract

HERE = Path(__file__).resolve().parent
API = "https://api.openalex.org/works"


def _get(params):
    with urllib.request.urlopen(f"{API}?{urllib.parse.urlencode(params)}") as r:
        return json.load(r)["results"]


def lookup(row):
    if row["doi"]:
        res = _get({"filter": f"doi:{row['doi']}"})
    else:
        title = row["title"].replace(",", " ").replace(":", " ")
        res = _get({"filter": f"title.search:{title}", "per-page": "3"})
        norm = lambda t: "".join(ch for ch in (t or "").lower() if ch.isalnum())
        res = [w for w in res if norm(w.get("title")) == norm(row["title"])]
    if not res:
        return None
    w = res[0]
    return {"openalex_id": w["id"], "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
            "abstract": _abstract(w.get("abstract_inverted_index"))}


def main():
    rows = list(csv.DictReader(open(HERE / "unicos.csv")))
    todo = [r for r in rows if len(r["abstract"]) < 500 or r["abstract"].rstrip().endswith("...")]
    out = {}
    for r in todo:
        hit = lookup(r)
        if hit:
            out[r["uid"]] = hit
        time.sleep(0.15)
    (HERE / "raw_enrich").mkdir(exist_ok=True)
    (HERE / "raw_enrich" / "abstracts.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    got = sum(1 for v in out.values() if len(v["abstract"]) > 500)
    print(f"to_enrich={len(todo)} found={len(out)} full_abstract={got}")


if __name__ == "__main__":
    main()
