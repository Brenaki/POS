"""Write relatorio/secoes/12b-estudos-incluidos.tex from incluidos.csv.

Usage: python gen_incluidos_tex.py

ABNT-style entries (up to three authors, then "et al."), sorted by first
author; authors come from unicos.csv in either Scopus ("Last, I. , Last, I.")
or IEEE/OpenAlex ("First Last; First Last") form.
"""

import csv
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "relatorio" / "secoes" / "12b-estudos-incluidos.tex"
TEX = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#", "$": r"\$",
       "–": "--", "—": "---", "Ş": r"\c{S}"}


def esc(s: str) -> str:
    return "".join(TEX.get(c, c) for c in s)


def abnt_name(n: str) -> str:
    n = n.strip()
    if "," in n:
        last, first = n.split(",", 1)
        return f"{last.strip().upper()}, {first.strip()}"
    parts = n.split()
    if len(parts) == 1:
        return parts[0].upper()
    initials = " ".join(p[0] + "." for p in parts[:-1] if p)
    return f"{parts[-1].upper()}, {initials}"


def authors(raw: str) -> str:
    raw = re.sub(r"\s*,\.\.\.\s*", " , ", raw)  # Scopus truncation marker
    if ";" in raw or "," not in raw:
        names = [a for a in raw.split(";") if a.strip()]
    else:
        names = [a for a in re.split(r"\s+,\s+", raw) if a.strip()]
    out = [abnt_name(a) for a in names[:3]]
    return "; ".join(out) + ("; et al." if len(names) > 3 else "")


def main():
    uni = {r["uid"]: r for r in csv.DictReader(open(HERE / "unicos.csv"))}
    # DOIs completed during the full-text phase (Scopus lists carry none).
    found = {t["uid"]: t["doi"] for t in json.loads((HERE / "fulltext_todo.json").read_text())}
    entries = []
    for r in csv.DictReader(open(HERE / "incluidos.csv")):
        u = uni[r["uid"]]
        source = re.sub(r"\s+,", ",", u["source"]).strip().rstrip(".")
        d = r["doi"] or found.get(r["uid"], "")
        doi = f" DOI: {esc(d)}." if d else ""
        entries.append(f"\\item {esc(authors(u['authors']))} {esc(u['title'].rstrip('.'))}. "
                       f"\\emph{{{esc(source)}}}, {u['year']}.{doi} "
                       f"[{r['uid']}, {r['fase2_codigo']}]")
    entries.sort(key=lambda e: e.lower())
    OUT.write_text("\\section*{Estudos incluídos na revisão sistemática}\n"
                   "\\addcontentsline{toc}{section}{Estudos incluídos na revisão sistemática}\n\n"
                   "{\\small Identificador do registro e código de inclusão entre colchetes "
                   "(I-a generaliza o Oracle; I-b Oracle como referência; I-c diversidade "
                   "$\\times$ acurácia). Gerado por "
                   "\\texttt{docs/revisao-sistematica/gen\\_incluidos\\_tex.py}.}\n\n"
                   "\\begin{description}\n" + "\n\n".join(entries) + "\n\\end{description}\n")
    print(f"{len(entries)} entries -> {OUT}")


if __name__ == "__main__":
    main()
