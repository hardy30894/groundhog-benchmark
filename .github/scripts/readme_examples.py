"""Writes the README's Python examples to files, so the workflow runs exactly what it shows."""

import re
from pathlib import Path

readme = Path("README.md").read_text()
blocks = re.findall(r"```python\n(.*?)```", readme, re.S)
cache = next(b for b in blocks if "class ExactCache" in b)
snippet = next(b for b in blocks if "data.instances" in b)
Path("mycache.py").write_text(cache)
Path("snippet.py").write_text(snippet)
Path("cmdcache.py").write_text(
    "import sys\n"
    "from mycache import ExactCache\n"
    "ExactCache(int(sys.argv[1]), sys.argv[2]).serve_forever()\n"
)
