"""Merge the raw search exports into the PRISMA record tables.

Usage: python prisma_merge.py

Reads raw/*.json (Scopus result lists, IEEE Xplore REST, OpenAlex API) and
writes:
  buscas.csv     one row per executed search: base, id, string, date, hits
  registros.csv  every retrieved record, with its origin (PRISMA "identified")
  unicos.csv     records after de-duplication (DOI, else normalised title),
                 with all origins joined; the input of the screening
Scopus result lists carry no DOI, so cross-base duplicates are matched on
the normalised title; the longest abstract among the duplicates is kept, and
abstracts still truncated or missing are completed from OpenAlex
(raw_enrich/abstracts.json, written by enrich_abstracts.py).
"""

import csv
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
# S3a/S3b are one Scopus search (S3) split by year only to get past the
# 200-rows-per-page limit of the result list: their records stand for S3,
# whose own export (first 200 of 224) is logged as the search but not merged.
SEARCH_ID = {"S3a": "S3", "S3b": "S3"}
PARTIAL = {("scopus", "S3")}


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())


def clean_abstract(a: str) -> str:
    a = re.sub(r"^Hide abstract\s*", "", a or "")
    a = re.sub(r"\s*(Opens in a new tab\.|View at Publisher|Related documents).*$", "", a)
    return re.sub(r"\s*©.*$", "", a).strip()


def load():
    searches, records = [], []
    for f in sorted(RAW.glob("*.json")):
        base, sid = f.stem.split("_", 1)
        data = json.loads(f.read_text())
        if sid not in SEARCH_ID:
            searches.append({"base": base, "search": sid, "query": data["query"],
                             "filters": data.get("filters", "in query"),
                             "retrieved": data["retrieved"][:10], "hits": data["total"],
                             "recovered": data["got"]})
        if (base, sid) in PARTIAL:
            continue
        for r in data["rows"]:
            records.append({
                "base": base, "search": SEARCH_ID.get(sid, sid),
                "record_id": r.get("eid") or r.get("id"), "doi": (r.get("doi") or "").lower(),
                "title": r["title"], "year": str(r.get("year") or ""),
                "authors": r.get("authors") or "", "source": r.get("source") or "",
                "type": r.get("type") or "", "abstract": clean_abstract(r.get("abstract", ""))})
    return searches, records


def fix_partial(searches, records):
    """Recovered count of a split search = records merged under its id."""
    for s in searches:
        if (s["base"], s["search"]) in PARTIAL:
            s["recovered"] = sum(1 for r in records
                                 if r["base"] == s["base"] and r["search"] == s["search"])


def dedupe(records):
    groups, by_doi, by_title = [], {}, {}
    for r in records:
        key_t = norm_title(r["title"])
        g = by_doi.get(r["doi"]) if r["doi"] else None
        g = g if g is not None else by_title.get(key_t)
        if g is None:
            g = {**r, "origins": set()}
            groups.append(g)
        g["origins"].add(f"{r['base']}:{r['search']}")
        for k in ("doi", "type"):
            if not g[k] and r[k]:
                g[k] = r[k]
        # IEEE's search endpoint truncates abstracts at ~400 chars ("..."):
        # keep the longest one among the duplicates.
        if len(r["abstract"]) > len(g["abstract"]):
            g["abstract"] = r["abstract"]
        if g["doi"]:
            by_doi[g["doi"]] = g
        by_title[key_t] = g
    for i, g in enumerate(groups, 1):
        g["uid"] = f"R{i:04d}"
        g["origins"] = "; ".join(sorted(g["origins"]))
    return groups


def apply_enrichment(groups):
    extra = {}
    # abstracts.json: OpenAlex by DOI/title; ieee_abstracts.json: the IEEE
    # document page, for IEEE-only records OpenAlex has no abstract for.
    for name in ("abstracts.json", "ieee_abstracts.json"):
        path = HERE / "raw_enrich" / name
        if path.exists():
            for uid, e in json.loads(path.read_text()).items():
                if len(e.get("abstract", "")) > len((extra.get(uid) or {}).get("abstract", "")):
                    extra[uid] = {**(extra.get(uid) or {}), **e}
    for g in groups:
        e = extra.get(g["uid"]) or {}
        if len(e.get("abstract", "")) > len(g["abstract"]):
            g["abstract"] = e["abstract"]
            g["abstract_source"] = "openalex"
        if not g["doi"] and e.get("doi"):
            g["doi"] = e["doi"]


def write(path, rows, cols):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    searches, records = load()
    fix_partial(searches, records)
    uniq = dedupe(records)
    apply_enrichment(uniq)
    write(HERE / "buscas.csv", searches,
          ["base", "search", "query", "filters", "retrieved", "hits", "recovered"])
    write(HERE / "registros.csv", records,
          ["base", "search", "record_id", "doi", "title", "year", "authors", "source", "type"])
    write(HERE / "unicos.csv", uniq,
          ["uid", "doi", "title", "year", "authors", "source", "type", "origins", "abstract"])
    print(f"searches={len(searches)} identified={len(records)} unique={len(uniq)} "
          f"duplicates={len(records) - len(uniq)} "
          f"no_abstract={sum(1 for g in uniq if len(g['abstract']) < 100)}")


if __name__ == "__main__":
    main()
