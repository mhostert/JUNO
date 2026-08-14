# Notebook sources

The three notebooks in the repository root are generated from the plain-Python cell lists
here, which makes them easy to diff and edit without wrestling with `.ipynb` JSON.

Each `nbN.py` defines `OUT` (the output notebook name) and `CELLS`, a list of
`("md" | "code", source)` tuples. `build_nb.py` assembles the notebook, executes it with
`nbclient`, writes it to the repository root, and reports any cell errors.

```bash
python notebook_sources/build_nb.py notebook_sources/nb0.py            # build and execute
python notebook_sources/build_nb.py notebook_sources/nb2.py --no-exec  # assemble only
```

Approximate execution times: `nb0` 4 s, `nb1` 25 s, `nb2` 150 s.

Editing a notebook directly in Jupyter is fine — the sources here are a convenience, not a
build requirement. If you do, either re-apply the change here or delete the corresponding
`nbN.py` so the two do not drift apart.

One gotcha: because each cell is a `r"""..."""` string, cell code cannot contain a triple
double-quote. Use `#` comments instead of docstrings inside notebook functions.
