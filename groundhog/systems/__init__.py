"""The seven systems measured in the paper, by the names the runner accepts."""

SYSTEMS = {
    "lexcache": "py:groundhog.systems.lexcache:LexCache",
    "gptcache": "py:groundhog.systems.gptcache:GPTCache",
    "vcache": "py:groundhog.systems.vcache:VCache",
    "w5h2": "py:groundhog.systems.w5h2:W5H2",
    "agentreuse": "py:groundhog.systems.agentreuse:AgentReuse",
    "agenticcache": "py:groundhog.systems.agenticcache:AgenticCache",
    "gencache": "py:groundhog.systems.gencache:GenCache",
}
