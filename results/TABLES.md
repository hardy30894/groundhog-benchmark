### Table 1: Seven systems at the settings of Table 1

| system | gross | earned | failed pairs | wrong writes | missed writes | wrong reads | wrong answers | served |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LexCache | 9.3% | 9.9% | 6/33 | 924 | 853 | 3,417 | 597 | 3,100 |
| GPTCache | 8.7% | 0.8% | 12/33 | 1,555 | 1,481 | 5,381 | 2,561 | 2,847 |
| vCache | 0.1% | 0.1% | 1/33 | 1 | 1 | 3 | 3 | 38 |
| W5H2 | 11.9% | 10.2% | 3/33 | 150 | 150 | 522 | 593 | 3,214 |
| AgentReuse | 70.7% | 11.7% | 11/33 | 945 | 979 | 522 | 1,230 | 7,443 |
| AgenticCache | -14.9% | 0.0% | 12/33 | 339 | 0 | 696 | 386 | 802 |
| GenCache | 93.8% | 22.2% | 5/33 | 257 | 859 | 412 | 1,053 | 7,127 |

### Table 3: The 11 families

| family | condition | turns per run | share of cost |
|---|---|---:|---:|
| lookup | identity | 2.0 | 2.0% |
| buried | identity | 2.0 | 7.9% |
| mixed | identity | 2.2 | 4.0% |
| chain | values | 3.0 | 3.9% |
| transform | values | 2.0 | 2.0% |
| judgement | meaning | 2.0 | 2.3% |
| reversal | meaning | 1.8 | 1.8% |
| drift | world | 2.0 | 2.1% |
| quiet | world | 3.0 | 4.2% |
| standing | world | 2.0 | 2.1% |
| horizon | depth | 14.4 | 67.7% |

### Table 5: LexCache by threshold

| threshold | failed pairs | wrong writes | missed writes | wrong reads | earned | gross |
|---|---:|---:|---:|---:|---:|---:|
| 0.7 | 12/33 | 1,052 | 966 | 3,417 | 9.8% | 9.5% |
| 0.8 | 6/33 | 924 | 853 | 3,417 | 9.9% | 9.3% |
| 0.9 | 3/33 | 150 | 150 | 150 | 9.9% | 10.7% |

### Table 6: GPTCache by similarity threshold

| threshold | failed pairs | wrong writes | missed writes | wrong reads | earned | gross |
|---|---:|---:|---:|---:|---:|---:|
| 0.80 | 13/33 | 1,587 | 1,524 | 5,535 | 0.9% | 9.3% |
| 0.85 | 12/33 | 1,555 | 1,481 | 5,381 | 0.8% | 8.7% |
| 0.90 | 12/33 | 1,461 | 1,390 | 5,017 | 0.7% | 7.8% |

### Table 7: vCache by error bound

| delta | failed pairs | wrong writes | missed writes | wrong reads | earned | gross |
|---|---:|---:|---:|---:|---:|---:|
| 0.005 | 2/33 | 2 | 4 | 14 | 0.1% | 0.1% |
| 0.01 | 1/33 | 1 | 1 | 3 | 0.1% | 0.1% |
| 0.02 | 8/33 | 13 | 9 | 43 | 0.2% | 0.1% |
| 0.05 | 8/33 | 18 | 15 | 68 | 0.3% | 0.4% |

### Table 8: W5H2 by risk tolerance

| alpha | failed pairs | wrong writes | missed writes | wrong reads | earned | gross |
|---|---:|---:|---:|---:|---:|---:|
| 0.05 | 2/33 | 5 | 5 | 522 | 9.8% | 11.2% |
| 0.10 | 3/33 | 150 | 150 | 522 | 10.2% | 11.9% |
| 0.20 | 5/33 | 157 | 157 | 522 | 10.3% | 12.1% |
| 0.30 | 5/33 | 157 | 157 | 522 | 10.3% | 12.1% |

### Table 9: AgentReuse by embedder and gamma, and four variants

| setting | failed pairs | wrong writes | missed writes | wrong reads | earned | gross |
|---|---:|---:|---:|---:|---:|---:|
| m3e-small, gamma 0.75 | 11/33 | 945 | 979 | 522 | 11.7% | 70.7% |
| m3e-small, gamma 0.80 | 11/33 | 943 | 977 | 522 | 11.7% | 70.7% |
| m3e-small, gamma 0.85 | 11/33 | 922 | 956 | 522 | 11.7% | 70.6% |
| m3e-small, gamma 0.90 | 11/33 | 885 | 919 | 519 | 11.8% | 70.6% |
| m3e-small, gamma 0.95 | 11/33 | 775 | 809 | 519 | 11.8% | 70.3% |
| MiniLM, gamma 0.75 | 8/33 | 747 | 781 | 519 | 12.2% | 70.6% |
| MiniLM, gamma 0.80 | 8/33 | 747 | 781 | 519 | 12.2% | 70.6% |
| MiniLM, gamma 0.85 | 8/33 | 747 | 781 | 519 | 12.2% | 70.6% |
| MiniLM, gamma 0.90 | 8/33 | 747 | 781 | 519 | 12.1% | 70.5% |
| MiniLM, gamma 0.95 | 8/33 | 747 | 781 | 519 | 11.9% | 70.3% |
| opening turn only | 6/33 | 348 | 348 | 522 | 9.6% | 11.9% |
| parameters left in the embedding | 11/33 | 945 | 979 | 522 | 11.7% | 70.7% |
| matched against all stored conversations | 11/33 | 945 | 979 | 522 | 11.2% | 70.2% |
| refusals labelled as the action they mention | 14/33 | 993 | 979 | 522 | 11.7% | 70.6% |

### Table 10: AgenticCache by K, with result fields, and two ablation arms

| setting | failed pairs | wrong writes | missed writes | wrong reads | earned | gross |
|---|---:|---:|---:|---:|---:|---:|
| K = 1 | 12/33 | 339 | 0 | 696 | 0.0% | -14.9% |
| K = 2 | 12/33 | 371 | 396 | 684 | 6.0% | 27.7% |
| K = 3 | 12/33 | 307 | 525 | 528 | 8.3% | 43.6% |
| K = 5 | 12/33 | 307 | 659 | 330 | 10.5% | 54.7% |
| K = 10 | 12/33 | 254 | 759 | 154 | 12.8% | 62.9% |
| K = 1, result fields as metadata | 6/33 | 99 | 0 | 75 | 0.0% | -0.4% |
| K = 2, result fields as metadata | 6/33 | 93 | 33 | 75 | 34.0% | 34.0% |
| K = 3, result fields as metadata | 6/33 | 93 | 39 | 75 | 44.9% | 45.0% |
| K = 5, result fields as metadata | 6/33 | 126 | 75 | 75 | 54.1% | 54.3% |
| K = 10, result fields as metadata | 6/33 | 123 | 75 | 75 | 61.6% | 61.9% |
| K = 1, checks update counts only | 12/33 | 246 | 729 | 75 | 0.0% | 15.0% |
| K = 1, checks replace only | 11/33 | 404 | 0 | 664 | 0.0% | -10.7% |

### Table 11: GenCache with and without the program writer's tokens

| setting | failed pairs | wrong writes | missed writes | wrong reads | earned | gross |
|---|---:|---:|---:|---:|---:|---:|
| program writing not counted | 5/33 | 257 | 859 | 412 | 22.2% | 93.8% |
| program writing counted | 5/33 | 257 | 859 | 412 | -8.2% | 63.4% |

### Table 12: Earned savings per family, three domains pooled, in percent

| family | LexCache | GPTCache | vCache | W5H2 | AgentReuse | AgenticCache | GenCache |
|---|---:|---:|---:|---:|---:|---:|---:|
| lookup | 35 | 0 | 0 | 36 | 36 | 0 | 38 |
| buried | 45 | 0 | 0 | 46 | 46 | 0 | 68 |
| mixed | 36 | 4 | 0 | 32 | 39 | 0 | 76 |
| chain | 19 | 0 | 0 | 19 | 52 | 0 | 93 |
| transform | 36 | 0 | 0 | 0 | 0 | 0 | 74 |
| judgement | 21 | 25 | 4 | 31 | 12 (3) | 0 | 60 |
| reversal | 37 | 3 (3) | 0 | 38 | 33 | 0 (3) | 91 |
| drift | 17 | 0 | 0 | 17 | 17 | 0 | 40 |
| quiet | 20 | 0 (3) | 0 (1) | 20 | 26 (3) | 0 (3) | 51 (2) |
| standing | 17 (3) | 0 (3) | 0 | 17 (3) | 17 (3) | 0 (3) | 45 (3) |
| horizon | 0 (3) | 0 (3) | 0 | 1 | 2 (2) | 0 (3) | 1 |

A number in parentheses is the count of domains with a wrong write on that family.

### Table 13: Earned savings per condition

| system | identity | values | meaning | world | depth |
|---|---:|---:|---:|---:|---:|
| LexCache | 41% | 25% | 28% | 19% (3) | 0% (3) |
| GPTCache | 1% | 0% | 15% (3) | 0% (6) | 0% (3) |
| vCache | 0% | 0% | 2% | 0% (1) | 0% |
| W5H2 | 41% | 13% | 35% | 19% (3) | 1% |
| AgentReuse | 43% | 34% | 22% (3) | 22% (6) | 2% (2) |
| AgenticCache | 0% | 0% | 0% (3) | 0% (6) | 0% (3) |
| GenCache | 66% | 87% | 73% | 47% (5) | 1% |

### Table 14: Earned savings per domain

| system | support | code | data |
|---|---:|---:|---:|
| LexCache | 9.9% | 9.8% | 9.9% |
| GPTCache | 0.8% | 0.7% | 0.8% |
| vCache | 0.1% | 0.1% | 0.1% |
| W5H2 | 10.3% | 10.1% | 10.2% |
| AgentReuse | 14.0% | 10.6% | 10.6% |
| AgenticCache | 0.0% | 0.0% | 0.0% |
| GenCache | 18.0% | 25.3% | 23.2% |

### Table 15: Earned savings by turns per run without a cache

| turns per run | 1.8 | 2.0 | 2.2 | 3.0 | 14.4 |
|---|---:|---:|---:|---:|---:|
| family-domain pairs | 3 | 18 | 3 | 6 | 3 |
| LexCache | 37.3 | 33.4 | 35.7 | 19.7 | 0.0 |
| GPTCache | 3.1 | 3.1 | 3.7 | 0.1 | 0.0 |
| vCache | 0.0 | 0.5 | 0.3 | 0.0 | 0.0 |
| W5H2 | 38.4 | 31.4 | 32.1 | 19.7 | 1.2 |
| AgentReuse | 32.8 | 29.0 | 39.0 | 38.5 | 1.6 |
| AgenticCache | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| GenCache | 90.5 | 58.7 | 76.1 | 71.5 | 1.3 |

