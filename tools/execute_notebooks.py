"""Execute both notebooks on a fresh kernel (restart + run all) and save them with outputs."""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ["01_exploration.ipynb", "02_system.ipynb"]


def main() -> int:
    for name in NOTEBOOKS:
        path = ROOT / "notebooks" / name
        nb = nbformat.read(path, as_version=4)
        for cell in nb.cells:
            if cell.cell_type == "code":
                cell.outputs, cell.execution_count = [], None
        client = NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}})
        client.execute()
        counts = [c.execution_count for c in nb.cells if c.cell_type == "code"]
        assert counts == list(range(1, len(counts) + 1)), f"{name}: execution counts not contiguous {counts}"
        nbformat.write(nb, path)
        print(f"executed {name}: {len(counts)} code cells, counts 1..{len(counts)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
