"""Fetch BibTeX from INSPIRE-HEP for every reference; write ref.bib.

For each entry: query INSPIRE by arXiv id when we have one, else by a
literal title search; take the top hit's BibTeX (INSPIRE's own key), and
record the mapping old-key -> INSPIRE-key so main.tex can be rewritten.
Entries INSPIRE does not have (software, NIST database, a 1984 KFA report,
the JUNO data release) fall back to hand-written entries.
"""
import json, re, sys, time, urllib.parse, urllib.request

REFS = {
 # key            : (arxiv-id or None, title fallback)
 "JUNO:2022_physics":     ("2104.02565", None),
 "JUNO:2020_TAO":         ("2005.08745", None),
 "DayaBay:2021_unfolded": ("2102.04614", None),
 "DayaBay:2017_evolution":("1704.01082", None),
 "Vogel:1999zy":          ("hep-ph/9903554", None),
 "Strumia:2003zx":        ("astro-ph/0302055", None),
 "Kopeikin:1997ve":       (None, "Inelastic scattering of tritium source antineutrinos on electrons of germanium atoms"),
 "Erler:2004in":          ("hep-ph/0409169", None),
 "Vinyoles:2016djt":      ("1611.09867", None),
 "Bahcall:1996qv":        ("nucl-th/9601044", "Standard neutrino spectrum from B-8 decay"),
 "Kopp:2013vaa":          ("1303.3011", None),
 "Berryman:2021yan":      ("2111.12530", None),
 "Barinov:2021asz":       ("2109.11482", None),
 "DANSS:2018fnn":         ("1804.04046", None),
 "PROSPECT:2020sxr":      ("2006.11210", None),
 "STEREO:2022nzk":        ("2210.07664", None),
 "RENO:2020hva":          ("2011.00896", None),
 "IsoDAR:2022":           ("2110.10635", None),
 "Dent:2019ueq":          ("1912.05733", None),
 "AristizabalSierra:2020rom": ("2010.15712", None),
 "AristizabalSierra:2025tao": ("2511.01812", None),
 "Aloni:2019ruo":         ("1903.03586", None),
 "Brodsky:1986mi":        (None, "Laser Induced Axion Photoproduction"),
 "Depta:2020wmr":         ("2002.08370", None),
 "Jaeckel:2006xm":        ("hep-ph/0610203", None),
 "Huber:2011wv":          ("1106.0687", None),
 "Mueller:2011nm":        ("1101.2663", None),
 "ParticleDataGroup:2024cfk": ("doi:10.1103/PhysRevD.110.030001", None),
 "Cowan:2010js":          ("1007.1727", None),
 "Minakata:2006gq":       ("hep-ph/0607284", None),
 "KATRIN:2025_sterile":   ("2503.18667", None),
 "JUNO:2025_solar":       ("2511.14593", None),
}

API = "https://inspirehep.net/api/literature"

def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "juno-writeup/1.0"})
    with urllib.request.urlopen(req, timeout=40) as r:
        return r.read().decode()

def search(q, n=3):
    url = f"{API}?q={urllib.parse.quote(q)}&size={n}&fields=texkeys,titles,arxiv_eprints,control_number"
    return json.loads(get(url))["hits"]["hits"]

def bibtex(recid):
    return get(f"{API}/{recid}?format=bibtex").strip()

out, mapping, missing = [], {}, []
for key, (arx, title) in REFS.items():
    hits = []
    try:
        if arx:
            hits = search(arx if arx.startswith("doi:") else f"arxiv:{arx}", 1)
        if not hits and title:
            hits = search(f't "{title}"', 3)
    except Exception as e:
        print(f"  !! {key}: query failed ({e})"); hits = []
    if not hits:
        missing.append(key); print(f"MISSING  {key}"); continue
    h = hits[0]; md = h["metadata"]
    recid = md["control_number"]
    try:
        bib = bibtex(recid)
    except Exception as e:
        missing.append(key); print(f"  !! {key}: bibtex fetch failed ({e})"); continue
    m = re.match(r'@\w+\{([^,]+),', bib)
    newkey = m.group(1) if m else key
    mapping[key] = newkey
    out.append(bib)
    ttl = md["titles"][0]["title"][:70]
    print(f"OK  {key:28s} -> {newkey:32s} | {ttl}")
    time.sleep(0.4)

# hand-written fallbacks for what INSPIRE cannot supply
FALLBACK = {
 "JUNO:2025_solar": '''@misc{JUNO:2025_solar,
  author = "{JUNO Collaboration}",
  title = "{First measurement of reactor antineutrino oscillations at JUNO: 59.1-day data release}",
  year = "2025",
  note = "Public data release accompanying the first solar-parameter result"
}''',
 "KATRIN:2025_sterile": '''@misc{KATRIN:2025_sterile,
  author = "{KATRIN Collaboration}",
  title = "{Search for eV-scale sterile neutrinos with the full KATRIN dataset}",
  year = "2025",
  note = "95\\% CL exclusion curve as released"
}''',
 "Bechteler:1984": '''@techreport{Bechteler:1984,
  author = "Bechteler, H. and Faissner, H. and Yogeshwar, R. and Seyfarth, H.",
  title = "{The spectrum of $\\gamma$ radiation emitted in the FRJ-1 (Merlin) reactor core and moderator region}",
  institution = "Institut f{\\"u}r Kernphysik, KFA J{\\"u}lich",
  year = "1984"
}''',
 "XCOM": '''@misc{XCOM,
  author = "Berger, M. J. and Hubbell, J. H. and Seltzer, S. M. and Chang, J. and Coursey, J. S. and Sukumar, R. and Zucker, D. S. and Olsen, K.",
  title = "{XCOM: Photon Cross Sections Database (version 1.5)}",
  howpublished = "NIST Standard Reference Database 8 (XGAM), \\url{https://physics.nist.gov/xcom}",
  year = "2010"
}''',
 "NEPTUNE": '''@misc{NEPTUNE,
  author = "Hostert, M.",
  title = "{NEPTUNE: neutrino--electron and trident scattering cross sections}",
  howpublished = "software package",
  year = "2026"
}''',
}
for k in ("Bechteler:1984", "XCOM", "NEPTUNE"):
    out.append(FALLBACK[k]); mapping[k] = k
for k in missing:
    if k in FALLBACK:
        out.append(FALLBACK[k]); mapping[k] = k
        print(f"FALLBACK {k}")

open("ref.bib", "w").write("\n\n".join(out) + "\n")
json.dump(mapping, open("keymap.json", "w"), indent=1)
print(f"\nwrote ref.bib with {len(out)} entries; {len(missing)} not on INSPIRE")
