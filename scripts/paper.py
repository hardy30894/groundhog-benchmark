"""Every configuration the paper reports: its result file, the table it appears in, and the
settings that reproduce it."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Run:
    name: str
    system: str
    settings: dict = field(default_factory=dict)
    tables: tuple = ()
    origin: str = ""

    @property
    def path(self):
        return f"results/{self.name}.json"


RUNS = [
    Run("control", "control", {}, ("3",), "repro-control"),
    *[
        Run(
            f"lexcache/threshold-{v}",
            "lexcache",
            {"GROUNDHOG_LEXCACHE_THRESHOLD": v},
            ("5",) + (("1",) if v == "0.8" else ()),
            f"trigram-words-{v}",
        )
        for v in ("0.7", "0.8", "0.9")
    ],
    Run(
        "gptcache/threshold-0.80",
        "gptcache",
        {"GROUNDHOG_GPTCACHE_THRESHOLD": "0.80"},
        ("6",),
        "wide-gptcache-0.80",
    ),
    Run(
        "gptcache/threshold-0.85",
        "gptcache",
        {"GROUNDHOG_GPTCACHE_THRESHOLD": "0.85"},
        ("1", "6"),
        "wide-gptcache",
    ),
    Run(
        "gptcache/threshold-0.90",
        "gptcache",
        {"GROUNDHOG_GPTCACHE_THRESHOLD": "0.90"},
        ("6",),
        "wide-gptcache-0.90",
    ),
    Run(
        "vcache/delta-0.005",
        "vcache",
        {"GROUNDHOG_VCACHE_DELTA": "0.005"},
        ("7",),
        "official-vcache-delta-0.005",
    ),
    Run(
        "vcache/delta-0.01",
        "vcache",
        {"GROUNDHOG_VCACHE_DELTA": "0.01"},
        ("1", "7"),
        "wide-official-vcache",
    ),
    Run(
        "vcache/delta-0.02",
        "vcache",
        {"GROUNDHOG_VCACHE_DELTA": "0.02"},
        ("7",),
        "official-vcache-delta-0.02",
    ),
    Run(
        "vcache/delta-0.05",
        "vcache",
        {"GROUNDHOG_VCACHE_DELTA": "0.05"},
        ("7",),
        "official-vcache-delta-0.05",
    ),
    Run(
        "w5h2/alpha-0.05",
        "w5h2",
        {"GROUNDHOG_W5H2_ALPHA": "0.05"},
        ("8",),
        "official-w5h2-alpha-0.05",
    ),
    Run(
        "w5h2/alpha-0.10",
        "w5h2",
        {"GROUNDHOG_W5H2_ALPHA": "0.10"},
        ("1", "8"),
        "wide-official-w5h2",
    ),
    Run(
        "w5h2/alpha-0.20",
        "w5h2",
        {"GROUNDHOG_W5H2_ALPHA": "0.20"},
        ("8",),
        "official-w5h2-alpha-0.20",
    ),
    Run(
        "w5h2/alpha-0.30",
        "w5h2",
        {"GROUNDHOG_W5H2_ALPHA": "0.30"},
        ("8",),
        "official-w5h2-alpha-0.30",
    ),
    *[
        Run(
            f"agentreuse/m3e-small-gamma-{g}",
            "agentreuse",
            {"GROUNDHOG_AGENTREUSE_EMBEDDER": "moka-ai/m3e-small", "GROUNDHOG_AGENTREUSE_GAMMA": g},
            ("9",) + (("1",) if g == "0.75" else ()),
            "wide-agentreuse" if g == "0.75" else f"agentreuse-gamma-{g}",
        )
        for g in ("0.75", "0.80", "0.85", "0.90", "0.95")
    ],
    *[
        Run(
            f"agentreuse/minilm-gamma-{g}",
            "agentreuse",
            {
                "GROUNDHOG_AGENTREUSE_EMBEDDER": "sentence-transformers/all-MiniLM-L6-v2",
                "GROUNDHOG_AGENTREUSE_GAMMA": g,
            },
            ("9",),
            f"agentreuse-minilm-gamma-{g}",
        )
        for g in ("0.75", "0.80", "0.85", "0.90", "0.95")
    ],
    Run(
        "agentreuse/opening-turn-only",
        "agentreuse",
        {"GROUNDHOG_AGENTREUSE_TURN0": "1"},
        ("9",),
        "agentreuse-turn0",
    ),
    Run(
        "agentreuse/parameters-left-in",
        "agentreuse",
        {"GROUNDHOG_AGENTREUSE_ABLATION": "withargs"},
        ("9",),
        "agentreuse-withargs",
    ),
    Run(
        "agentreuse/matched-against-all",
        "agentreuse",
        {"GROUNDHOG_AGENTREUSE_ABLATION": "oneintent"},
        ("9",),
        "agentreuse-oneintent",
    ),
    Run(
        "agentreuse/refusals-labelled-by-action",
        "agentreuse",
        {"GROUNDHOG_AGENTREUSE_LABELS": "semantic"},
        ("9",),
        "agentreuse-semantic",
    ),
    *[
        Run(
            f"agenticcache/k{k}",
            "agenticcache",
            {"GROUNDHOG_AGENTICCACHE_K": k, "GROUNDHOG_AGENTICCACHE_SEED": "v0"},
            ("10",) + (("1",) if k == "1" else ()),
            "wide-agenticcache" if k == "1" else f"agenticcache-k{k}-v0",
        )
        for k in ("1", "2", "3", "5", "10")
    ],
    *[
        Run(
            f"agenticcache/fields-k{k}",
            "agenticcache",
            {
                "GROUNDHOG_AGENTICCACHE_K": k,
                "GROUNDHOG_AGENTICCACHE_METADATA": "fields",
                "GROUNDHOG_AGENTICCACHE_SEED": "v0",
            },
            ("10",),
            f"agenticcache-fields-k{k}",
        )
        for k in ("1", "2", "3", "5", "10")
    ],
    Run(
        "agenticcache/updates-only",
        "agenticcache",
        {
            "GROUNDHOG_AGENTICCACHE_K": "1",
            "GROUNDHOG_AGENTICCACHE_ARM": "updates",
            "GROUNDHOG_AGENTICCACHE_SEED": "v0",
        },
        ("10",),
        "agenticcache-updates-k1",
    ),
    Run(
        "agenticcache/replacement-only",
        "agenticcache",
        {
            "GROUNDHOG_AGENTICCACHE_K": "1",
            "GROUNDHOG_AGENTICCACHE_ARM": "replacement",
            "GROUNDHOG_AGENTICCACHE_SEED": "v0",
        },
        ("10",),
        "agenticcache-replacement-k1",
    ),
    Run(
        "gencache/gpt-5-mini",
        "gencache",
        {"GROUNDHOG_GENCACHE_MODEL": "openai/gpt-5-mini"},
        ("1", "11"),
        "wide11-gencache",
    ),
]

# The configuration each system is reported at in Table 1.
TABLE1 = {
    "LexCache": "lexcache/threshold-0.8",
    "GPTCache": "gptcache/threshold-0.85",
    "vCache": "vcache/delta-0.01",
    "W5H2": "w5h2/alpha-0.10",
    "AgentReuse": "agentreuse/m3e-small-gamma-0.75",
    "AgenticCache": "agenticcache/k1",
    "GenCache": "gencache/gpt-5-mini",
}

BY_NAME = {run.name: run for run in RUNS}
