# Dependencies by system

The harness, LexCache and AgenticCache need nothing beyond the Python standard library.
Each other system needs the packages in its file here, pinned to the versions its published
result file recorded. Install one file per virtual environment:

```bash
python -m venv .venv-gptcache && . .venv-gptcache/bin/activate
pip install -e . -r requirements/gptcache.txt
```

| system | file | Python the published run used |
|---|---|---|
| GPTCache | `gptcache.txt` | 3.12 |
| vCache | `vcache.txt` | 3.12 |
| W5H2 | `w5h2.txt`, plus the authors' checkout (see the file) | 3.10 |
| AgentReuse | `agentreuse.txt` | 3.12 |
| GenCache | `gencache.txt`, plus `OPENROUTER_API_KEY` | 3.12 |
