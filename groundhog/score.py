"""Scoring. A wrong write fails the family-domain pair whatever the system saved; savings
count only on instances that came out right."""

import collections

from .world import WRITES

EDGE = " \t\n,.;:!?\"'()[]{}"


def names_in(text):
    """Tokens that contain both a letter and a digit, which is how identifiers look here."""
    for word in text.split():
        word = word.strip(EDGE)
        if any(c.isdigit() for c in word) and any(c.isalpha() for c in word):
            yield word


def notes_name_elsewhere(want, got):
    """Identifiers a run's free-text notes name that its required calls never touch."""
    mine = {
        value.lower()
        for effect in want["effects"]
        for _, value in effect[1]
        if isinstance(value, str)
    }
    strayed = []
    for said in got.get("notes") or ():
        for named in names_in(said):
            if named.lower() not in mine:
                strayed.append(named)
    return strayed


def answer_lacks(want, got):
    """What the final answer is missing, and anything it says that it must not."""
    wanted = want["answer"]
    pieces = [wanted] if isinstance(wanted, str) else list(wanted)
    have = got["answer"].lower()
    missing = [piece for piece in pieces if piece and piece.lower() not in have]
    contradicted = [
        f"also says {other}"
        for other in want.get("excludes") or ()
        if other and other.lower() in have
    ]
    return missing + contradicted


def verify_control(family, oracle, control_runs):
    """The benchmark checking itself: the run without a cache must satisfy every instance."""
    problems = []
    for i, (want, got) in enumerate(zip(oracle, control_runs)):
        same = (
            got["effects"] == want["effects"]
            if want.get("ordered")
            else collections.Counter(got["effects"]) == collections.Counter(want["effects"])
        )
        if not same:
            problems.append(
                f"{family} run {i}: control effects {got['effects']} != oracle {want['effects']}"
            )
        missing = answer_lacks(want, got)
        if missing:
            problems.append(f"{family} run {i}: control answer {got['answer']!r} lacks {missing!r}")
    return problems


def score(family, oracle, control_runs, system_runs, control_cost, system_cost, served_metered):
    """One row: error counts, gross and earned savings, and how much reuse happened."""
    wrong_reads = 0
    wrong_writes = 0
    wrong_answers = 0
    missed_writes = 0
    stale_notes = 0
    notes_changed = 0
    earned = 0
    for want, control, got in zip(oracle, control_runs, system_runs):
        before = wrong_reads + wrong_writes + wrong_answers
        # Compared with multiplicity: doing a required call twice is one call nobody asked for.
        required = collections.Counter(want["effects"])
        did = collections.Counter(got["effects"])
        skipped = list((required - did).elements())
        missed_writes += sum(1 for e in skipped if e[0] in WRITES)
        for effect, times in (did - required).items():
            if effect[0] in WRITES:
                wrong_writes += times
            else:
                wrong_reads += times
        if answer_lacks(want, got):
            wrong_answers += 1
        elif want.get("ordered") and got["effects"] != want["effects"]:
            wrong_answers += 1
        # Reported, never scored.
        stale_notes += len(notes_name_elsewhere(want, got))
        if sorted(got.get("notes") or ()) != sorted(control.get("notes") or ()):
            notes_changed += 1
        # An instance earns its saving only with nothing wrong and nothing left undone.
        if wrong_reads + wrong_writes + wrong_answers == before and not skipped:
            earned += control.get("metered", 0) - got.get("metered", 0)
    served = served_metered
    first = next((i + 1 for i, r in enumerate(system_runs) if r["served"]), None)
    prefixes = [leading_free(r) for r in system_runs]
    depths = [len(r.get("free") or []) for r in system_runs]
    return {
        "family": family,
        "gate": "FAIL" if wrong_writes else "pass",
        "wrong_writes": wrong_writes,
        "wrong_reads": wrong_reads,
        "wrong_answers": wrong_answers,
        "missed_writes": missed_writes,
        "stale_notes": stale_notes,
        "notes_changed": notes_changed,
        "control_tokens": control_cost,
        "system_tokens": system_cost,
        "saved": (control_cost - system_cost) / control_cost if control_cost else 0.0,
        "earned": earned / control_cost if control_cost else 0.0,
        "served_turns": served,
        "first_saving_at_run": first,
        "steps_per_run": round(sum(depths) / len(depths), 1) if depths else 0.0,
        "served_prefix": round(sum(prefixes) / len(prefixes), 1) if prefixes else 0.0,
        "checked_turns": sum(r.get("checked", 0) for r in system_runs),
    }


def leading_free(run):
    """Turns served free from the start of a run, before the first one that cost anything."""
    free = run.get("free") or []
    count = 0
    for was_free in free:
        if not was_free:
            break
        count += 1
    return count


def table(rows):
    header = (
        f"  {'family':18} {'gate':8} {'wrong-writes':12} {'wrong-reads':11} "
        f"{'wrong-answers':13} {'missed':>7} {'saved':>6} {'earned':>7} {'net-served':>10} "
        f"{'first-free':>10} {'steps':>6} {'prefix':>7}"
    )
    lines = [header]
    for r in rows:
        first = r["first_saving_at_run"] if r["first_saving_at_run"] else "-"
        lines.append(
            f"  {r['family']:18} {r['gate']:8} {r['wrong_writes']:12} {r['wrong_reads']:11} "
            f"{r.get('wrong_answers', 0):13} {r.get('missed_writes', 0):7} "
            f"{r['saved']:6.0%} {r.get('earned', 0):7.0%} "
            f"{r['served_turns']:10} {first:>10} "
            f"{r.get('steps_per_run', 0):6.1f} {r.get('served_prefix', 0):7.1f}"
        )
    for r in rows:
        stale = r.get("stale_notes", 0)
        if stale:
            lines.append(
                f"  {r['family']:18} left {stale} note(s) naming a subject the run "
                f"never touched. Reported, not scored."
            )
    return "\n".join(lines)
