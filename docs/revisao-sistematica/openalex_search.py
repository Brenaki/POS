"""OpenAlex leg of the PRISMA search (open database, reproducible by API).

Usage: python openalex_search.py S1 S2 S3 S4
Runs each queries/ieee_<S>.txt boolean string (the Scopus string without its
field wrapper and filters) against title+abstract, years 2021-2026, languages
en/pt/zh, excluding type=preprint, and writes raw/openalex_<S>.json.
OpenAlex rejects wildcards inside quoted phrases and stems terms itself, so
`*` is dropped; boolean queries with >5 operators are limited to 1 req/s.
"""

import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
API = "https://api.openalex.org/works"
FILTERS = ("publication_year:2021-2026,language:en|pt|zh,type:!preprint")


def _abstract(inv):
    if not inv:
        return ""
    pos = sorted((i, w) for w, idx in inv.items() for i in idx)
    return " ".join(w for _, w in pos)


def search(query: str) -> dict:
    rows, cursor, total = [], "*", None
    while cursor:
        params = {"filter": f"title_and_abstract.search:{query},{FILTERS}",
                  "per-page": "200", "cursor": cursor}
        with urllib.request.urlopen(f"{API}?{urllib.parse.urlencode(params)}") as r:
            j = json.load(r)
        total = j["meta"]["count"]
        for w in j["results"]:
            rows.append({"id": w["id"], "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
                         "title": w.get("title") or "", "year": w.get("publication_year"),
                         "type": w.get("type"), "language": w.get("language"),
                         "source": ((w.get("primary_location") or {}).get("source") or {}).get("display_name", ""),
                         "authors": "; ".join(a["author"]["display_name"] for a in w.get("authorships", [])),
                         "cited": w.get("cited_by_count"),
                         "abstract": _abstract(w.get("abstract_inverted_index"))})
        cursor = j["meta"].get("next_cursor") if j["results"] else None
        time.sleep(1.2)
    return {"source": "OpenAlex", "query": query, "filters": FILTERS, "total": total,
            "got": len(rows), "retrieved": datetime.now(timezone.utc).isoformat(), "rows": rows}


if __name__ == "__main__":
    for s in sys.argv[1:]:
        q = (HERE / "queries" / f"ieee_{s}.txt").read_text().strip().replace("*", "")
        data = search(q)
        (HERE / "raw" / f"openalex_{s}.json").write_text(json.dumps(data, indent=1, ensure_ascii=False))
        print(s, data["total"], data["got"])
