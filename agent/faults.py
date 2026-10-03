"""Eight fault injectors, grouped by the step they corrupt. Each fn(output, input) -> corrupted output.
Faults are SUBTLE on purpose: no 'UNKNOWN'/'NOT_FOUND' giveaway markers, outputs always look plausible.
step_idx: 0 plan, 1 retrieve, 2 reason, 3 tool, 4 answer."""
import random, sys, os
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.agent import DOCS, CITIES, CITY_INFO, COMPANIES, COUNTRIES

DEFAULT_CITY = Counter(COMPANIES.values()).most_common(1)[0][0]   # the "lazy guess" when context is lost

def _other(options, current):
    return random.choice([o for o in options if o != current])

# ---- step 0: plan ----
def _bad_plan(o, i):                 # plan picks a different (real) company
    return {**o, "company": _other(list(COMPANIES), o["company"])}

# ---- step 1: retrieve ----
def _irrelevant_retrieval(o, i):     # retrieves a convincing set of docs... about the WRONG company
    other = _other(list(COMPANIES), i["company"])
    third = o["docs"][2] if len(o["docs"]) > 2 else DOCS[-1]
    return {"docs": [f"{other} is headquartered in {COMPANIES[other]}.",
                     f"{other} was founded by a team of engineers and sells software.", third]}

# ---- step 2: reason ----
def _dropped_context(o, i):          # model ignores the docs and falls back to its most common guess
    return {"city": DEFAULT_CITY}

def _corrupted_state(o, i):          # city gets swapped for another real city
    return {"city": _other(CITIES, o["city"])}

# ---- step 3: tool ----
def _wrong_tool_arg(o, i):           # tool answers for the wrong city (value is real, just not ours)
    key, v = i["attr"], o["value"]
    return {"value": random.choice([CITY_INFO[c][key] for c in CITIES if CITY_INFO[c][key] != v])}

def _hallucinated_tool_output(o, i): # tool "returns" a plausible but invented value
    v = o["value"]
    if isinstance(v, int):
        while True:
            nv = int(v * random.choice([0.9, 1.1, 1.2])) + random.randint(1, 999)
            if nv != v: return {"value": nv}
    return {"value": _other(COUNTRIES, v)}

# ---- step 4: answer ----
def _answer_format_error(o, i):      # right value, wrong format ("120,000" / "zandria")
    a = str(o["answer"])
    if a.isdigit(): return {"answer": f"{int(a):,}" if int(a) >= 1000 else a + ".0"}
    return {"answer": a.lower()}

def _answer_wrong_rounding(o, i):    # value rounded / truncated on the way out
    a = str(o["answer"])
    if a.isdigit():
        r = str(int(float(f"{int(a):.2g}")))
        return {"answer": r if r != a else str(int(a) + 10)}
    return {"answer": a[:-1]}

FAULTS_BY_STEP = {
    0: {"bad_plan": _bad_plan},
    1: {"irrelevant_retrieval": _irrelevant_retrieval},
    2: {"dropped_context": _dropped_context, "corrupted_state": _corrupted_state},
    3: {"wrong_tool_arg": _wrong_tool_arg, "hallucinated_tool_output": _hallucinated_tool_output},
    4: {"answer_format_error": _answer_format_error, "answer_wrong_rounding": _answer_wrong_rounding},
}
FAULTS = {name: {"step_idx": step, "fn": fn} for step, d in FAULTS_BY_STEP.items() for name, fn in d.items()}

def make_fault(name):
    return {**FAULTS[name], "type": name}