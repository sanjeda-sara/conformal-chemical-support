"""Check every reference cited in main.tex and supplement.tex against the public record, and write the citation
contexts needed to check that each source supports the sentence that cites it.

Run on a machine with internet access (it queries Crossref, arXiv and the reference's own URL):
    python scripts/check_references.py
Outputs
    tables/reference_check.csv     one row per cited key: status OK or CHECK, reasons, what the registry says
    docs/citation_contexts.md      for each key: the reference as printed, and every sentence that cites it
                                   (numbers filled in); this is the file to give a reviewer for claim checks
What it verifies automatically: the DOI resolves and its title, first author and year match the bib entry; arXiv IDs
resolve and match; PMLR/ICLR/other URLs load and show the title.  What it cannot verify: whether the cited work says
what our sentence says.  That is the manual part, using docs/citation_contexts.md.
"""
import difflib
import html
import json
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
TEX = REPO / "paper/latex"
UA = {"User-Agent": "recomb-reference-check/1.0 (python urllib)"}


# ----------------------------------------------------------------------------- parsing
def parse_bib(path):
    s = Path(path).read_text()
    out = {}
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", s):
        typ, key, i = m.group(1).lower(), m.group(2), m.end()
        depth, j = 1, i
        while depth and j < len(s):
            depth += {"{": 1, "}": -1}.get(s[j], 0); j += 1
        body = s[i:j - 1]
        f, k = {}, 0
        while True:
            mm = re.compile(r"\s*(\w+)\s*=\s*").match(body, k)
            if not mm:
                break
            name, k = mm.group(1).lower(), mm.end()
            if body[k] == "{":
                d, e = 1, k + 1
                while d:
                    d += {"{": 1, "}": -1}.get(body[e], 0); e += 1
                val, k = body[k + 1:e - 1], e
            elif body[k] == '"':
                e = body.index('"', k + 1); val, k = body[k + 1:e], e + 1
            else:
                e = re.compile(r"[,\n]").search(body, k); e = e.start() if e else len(body); val, k = body[k:e].strip(), e
            f[name] = val
            k = body.find(",", k) + 1 if body.find(",", k) >= 0 else len(body)
        f["_type"] = typ
        out[key] = f
    return out


def cited_keys():
    keys = {}
    for fn in ("main.tex", "supplement.tex"):
        for m in re.finditer(r"\\cite[a-z]*\{([^}]*)\}", (TEX / fn).read_text()):
            for k in m.group(1).split(","):
                keys.setdefault(k.strip(), set()).add(fn)
    return keys


def detex(t):
    t = re.sub(r"\\['`^\"~=.uvHtcdb]\{?\\?([a-zA-Z])\}?", r"\1", t)
    t = t.replace("{", "").replace("}", "").replace("\\&", "&").replace("--", "-")
    return re.sub(r"\s+", " ", t).strip()


def norm(t):
    t = unicodedata.normalize("NFKD", detex(t or "")).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def sim(a, b):
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def first_family(authors):
    a = detex(authors).split(" and ")[0].strip()
    return norm(a.split(",")[0] if "," in a else a.split()[-1])


# ----------------------------------------------------------------------------- lookups
def get(url, as_json=False, timeout=30):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read().decode("utf-8", "replace")
        return (json.loads(data) if as_json else data), r.status, r.geturl()


def crossref_doi(doi):
    j, _, _ = get("https://api.crossref.org/works/" + urllib.parse.quote(doi), as_json=True)
    m = j["message"]
    yr = None
    for k in ("published-print", "published-online", "issued"):
        if m.get(k, {}).get("date-parts"):
            yr = m[k]["date-parts"][0][0]; break
    au = m.get("author", [{}])
    return {"title": (m.get("title") or [""])[0], "first_author": au[0].get("family", "") if au else "",
            "year": yr, "venue": (m.get("container-title") or [""])[0]}


def crossref_search(f):
    q = urllib.parse.quote(detex(f.get("title", "")) + " " + detex(f.get("author", "")).split(" and ")[0])
    j, _, _ = get(f"https://api.crossref.org/works?query.bibliographic={q}&rows=1", as_json=True)
    it = j["message"]["items"]
    if not it:
        return None
    m = it[0]; au = m.get("author", [{}])
    yr = (m.get("issued", {}).get("date-parts") or [[None]])[0][0]
    return {"title": (m.get("title") or [""])[0], "first_author": au[0].get("family", "") if au else "", "year": yr,
            "venue": (m.get("container-title") or [""])[0], "doi": m.get("DOI", "")}


def arxiv(aid):
    x, _, _ = get(f"https://export.arxiv.org/api/query?id_list={aid}")
    ent = x.split("<entry>")[1] if "<entry>" in x else ""
    t = re.search(r"<title>(.*?)</title>", ent, re.S); a = re.search(r"<name>(.*?)</name>", ent)
    y = re.search(r"<published>(\d{4})", ent)
    return {"title": html.unescape(t.group(1)) if t else "", "first_author": a.group(1).split()[-1] if a else "",
            "year": int(y.group(1)) if y else None, "venue": "arXiv"}


def page(url, title):
    x, st, final = get(url)
    words = [w for w in norm(title).split() if len(w) > 3]
    hit = sum(w in norm(x) for w in words) / max(len(words), 1)
    return {"title": f"[page loads; {hit:.0%} of title words found]", "first_author": "", "year": None,
            "venue": final, "_hit": hit}


# ----------------------------------------------------------------------------- main
def main():
    bib = parse_bib(TEX / "references_final.bib")
    keys = cited_keys()
    rows = []
    for key in sorted(keys):
        f = bib.get(key)
        r = {"key": key, "cited_in": ",".join(sorted(keys[key])), "status": "CHECK", "reasons": "", "bib_title": "",
             "bib_year": "", "bib_first_author": "", "found_title": "", "found_year": "", "found_first_author": "",
             "found_venue": "", "source": ""}
        if f is None:
            r["reasons"] = "cited but not in references_final.bib"; rows.append(r); continue
        r.update(bib_title=detex(f.get("title", "")), bib_year=f.get("year", ""), bib_first_author=first_family(f.get("author", "")))
        why = []
        au = [a.strip() for a in detex(f.get("author", "")).split(" and ")]
        if len(au) != len(set(au)):
            why.append("duplicate author name in bib entry")
        aid = re.search(r"(\d{4}\.\d{4,5})", f.get("eprint", "") + " " + f.get("howpublished", "") + " " + f.get("url", ""))
        try:
            if f.get("doi"):
                got = crossref_doi(f["doi"]); r["source"] = "crossref doi"
            elif aid:
                got = arxiv(aid.group(1)); r["source"] = "arxiv"
            elif f.get("url"):
                got = page(f["url"], f.get("title", "")); r["source"] = "url"
            else:
                got = crossref_search(f) or {}; r["source"] = "crossref search (no identifier in bib)"
                why.append("no DOI/arXiv/URL in bib entry; top search hit shown")
        except Exception as e:                                   # network or HTTP error
            got = {}; why.append(f"lookup failed: {type(e).__name__}: {str(e)[:80]}")
        time.sleep(0.4)
        r.update(found_title=got.get("title", ""), found_year=got.get("year", ""), found_first_author=got.get("first_author", ""),
                 found_venue=got.get("venue", ""))
        if got:
            if r["source"] == "url":
                if got.get("_hit", 0) < 0.8:
                    why.append("title not clearly on the page")
            else:
                s_ = sim(r["bib_title"], got.get("title", ""))
                if s_ < 0.90:
                    why.append(f"title similarity {s_:.2f}")
                if norm(got.get("first_author", "")) and norm(got["first_author"]) != r["bib_first_author"]:
                    why.append(f"first author: bib '{r['bib_first_author']}' vs '{got['first_author']}'")
                try:
                    dy = int(r["bib_year"]) - int(got.get("year"))
                    if dy:
                        why.append(f"year: bib {r['bib_year']} vs {got.get('year')}" + (" (online vs print?)" if abs(dy) == 1 else ""))
                except (TypeError, ValueError):
                    pass
        r["reasons"] = "; ".join(why)
        r["status"] = "OK" if not why else "CHECK"
        rows.append(r)
    R = pd.DataFrame(rows)
    R.to_csv(REPO / "tables/reference_check.csv", index=False)
    write_contexts(bib, keys, R)
    pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 90)
    print(R[["key", "status", "source", "reasons"]].to_string(index=False))
    print(f"\n{(R.status == 'OK').sum()} OK, {(R.status == 'CHECK').sum()} to check by hand.  "
          f"Details: tables/reference_check.csv.  Claim checks: docs/citation_contexts.md")


def write_contexts(bib, keys, R):
    num = {}
    for line in (TEX / "generated/numbers.tex").read_text().splitlines():
        m = re.match(r"\\expandafter\\def\\csname v@(.*?)\\endcsname\{(.*)\}$", line)
        if m:
            num[m.group(1)] = m.group(2)
    txt = {}
    for fn in ("main.tex", "supplement.tex"):
        s = (TEX / fn).read_text()
        s = re.sub(r"(?<!\\)%.*", "", s)
        s = re.sub(r"\\V\{([^}]*)\}", lambda m: num.get(m.group(1), "??" + m.group(1)), s)
        txt[fn] = s
    st = dict(zip(R.key, R.status))
    out = ["# Citation contexts", "",
           "For each cited reference: the entry as it appears in the paper's bibliography, the automatic check, and every "
           "sentence that cites it. Check that the source supports each sentence as written.", ""]
    for key in sorted(keys):
        f = bib.get(key, {})
        ref = "; ".join(x for x in (detex(f.get("author", "")), detex(f.get("title", "")),
                                    detex(f.get("journal", f.get("booktitle", f.get("publisher", f.get("howpublished", ""))))),
                                    f.get("year", ""), ("doi:" + f["doi"]) if f.get("doi") else f.get("url", "")) if x)
        out += [f"## {key}  [{st.get(key, '?')}]", "", ref, ""]
        for fn, s in txt.items():
            for m in re.finditer(r"\\cite[a-z]*\{([^}]*)\}", s):
                if key not in [k.strip() for k in m.group(1).split(",")]:
                    continue
                ends = [e.end() for e in re.finditer(r"\.\s|\n\s*\n", s[:m.start()])]
                a = ends[-1] if ends else 0
                nxt = re.compile(r"\.(\s|$)").search(s, m.end()); b = nxt.start() + 1 if nxt else len(s)
                sent = re.sub(r"\s+", " ", s[a:b]).strip()
                sent = re.sub(r"\\cite[a-z]*\{([^}]*)\}", r"[\1]", sent)
                out.append(f"- *{fn}*: {sent}")
        out.append("")
    (REPO / "docs/citation_contexts.md").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
