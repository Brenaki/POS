"""List the records that go to full-text eligibility (phase 2).

Usage: python fulltext_prep.py
Takes INCLUDE and MAYBE from triagem.csv, completes missing DOIs from
OpenAlex (exact normalised-title match) and IEEE record ids, and writes
fulltext_todo.json, the input of fulltext_check.js.
"""

import csv
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def norm(t):
    return "".join(c for c in (t or "").lower() if c.isalnum())


def doi_by_title(title):
    q = title.replace(",", " ").replace(":", " ")
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(
        {"filter": f"title.search:{q}", "per-page": "5"})
    with urllib.request.urlopen(url) as r:
        res = json.load(r)["results"]
    for w in res:
        if norm(w.get("title")) == norm(title) and w.get("doi"):
            return w["doi"].replace("https://doi.org/", "")
    return ""


def main():
    ieee = {}
    for r in csv.DictReader(open(HERE / "registros.csv")):
        if r["base"] == "ieee":
            ieee[norm(r["title"])] = r["record_id"]
    todo = []
    for r in csv.DictReader(open(HERE / "triagem.csv")):
        if r["fase1_decisao"] == "EXCLUDE":
            continue
        doi = r["doi"] or doi_by_title(r["title"])
        time.sleep(0.15)
        todo.append({"uid": r["uid"], "doi": doi, "ieee": ieee.get(norm(r["title"]), ""),
                     "title": r["title"], "fase1": r["fase1_decisao"]})
    (HERE / "fulltext_todo.json").write_text(json.dumps(todo, indent=1, ensure_ascii=False))
    print(len(todo), "no locator:", [t["uid"] for t in todo if not t["doi"] and not t["ieee"]])


if __name__ == "__main__":
    main()
