"""Build and execute a notebook from a python module defining CELLS."""

import importlib.util
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path("/Users/mhostert/Repos/JUNO")


def build(cells, out_path, execute=True, timeout=1800):
    nb = nbformat.v4.new_notebook()
    nb.metadata.update(
        {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": sys.version.split()[0]},
        }
    )
    for kind, src in cells:
        src = src.strip("\n")
        if kind == "md":
            nb.cells.append(nbformat.v4.new_markdown_cell(src))
        else:
            nb.cells.append(nbformat.v4.new_code_cell(src))

    out_path = Path(out_path)
    if execute:
        t0 = time.time()
        client = NotebookClient(
            nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}},
            allow_errors=True,
        )
        client.execute()
        print(f"executed in {time.time() - t0:.1f}s")

    nbformat.write(nb, out_path)

    # Report errors
    n_err = 0
    for i, cell in enumerate(nb.cells):
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                n_err += 1
                print(f"\n=== ERROR in cell {i} ===")
                print(cell.source[:400])
                print("---")
                print("\n".join(output.get("traceback", []))[-2500:])
    print(f"\n{n_err} error(s); wrote {out_path}")
    return n_err


def main():
    mod_path = Path(sys.argv[1])
    spec = importlib.util.spec_from_file_location("nbcells", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    execute = "--no-exec" not in sys.argv
    sys.exit(build(mod.CELLS, ROOT / mod.OUT, execute=execute))


if __name__ == "__main__":
    main()
