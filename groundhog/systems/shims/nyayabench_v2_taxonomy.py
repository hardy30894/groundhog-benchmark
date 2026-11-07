"""The module W5H2's released scripts import but the release does not ship.

hierarchical.py in github.com/nabaos/w5h2-intent-cache imports INTENT_TO_W5H2 and
W5H2_CLASSES from nyayabench_v2_taxonomy; the release defines them in w5h2_taxonomy.py.
"""

from w5h2_taxonomy import INTENT_TO_W5H2, W5H2_CLASSES  # noqa: F401
