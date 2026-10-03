"""Six fault injectors. Each corrupts ONE step's output AFTER it is computed (so the step before it was fine
and the injected step is the true culprit). step_idx: 0 plan, 1 retrieve, 2 reason, 3 tool, 4 answer."""
import random, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.agent import DOCS, CITIES, COMPANIES, COUNTRIES

def _other(options, current):
    return random.choice([o for o in options if o != current])

def _bad_plan(o):            # step 0: plan picks the wrong company
    return {**o, "company": _other(list(COMPANIES), o["company"])}

def _irrelevant_retrieval(o):  # step 1: retrieves unrelated docs
    return {"docs": random.sample([d for d in DOCS if d not in o["docs"]], 3)}

def _dropped_context(o):     # step 2: model "forgets" the docs, can't find the city
    return {"city": "UNKNOWN"}

def _corrupted_state(o):     # step 2: city name gets corrupted (SUBTLE: looks almost right)
    return {"city": o["city"][:-2]}

def _wrong_tool_arg(o):      # step 3: tool called for the wrong city (plausible value, wrong source)
    from agent.agent import CITY_INFO
    v = o["value"]
    key = "population" if isinstance(v, int) else "country"
    alts = [CITY_INFO[c][key] for c in CITIES if CITY_INFO[c][key] != v]
    return {"value": random.choice(alts)}

def _hallucinated_tool_output(o):  # step 3: tool "returns" an invented value (SUBTLE: close to the truth)
    v = o["value"]
    if isinstance(v, int):
        return {"value": int(v * random.choice([0.9, 1.1, 1.2])) + random.randint(1, 999)}
    return {"value": "Zandria" if v != "Zandria" else "Molvia"}

FAULTS = {
    "bad_plan":                 {"step_idx": 0, "fn": _bad_plan},
    "irrelevant_retrieval":     {"step_idx": 1, "fn": _irrelevant_retrieval},
    "dropped_context":          {"step_idx": 2, "fn": _dropped_context},
    "corrupted_state":          {"step_idx": 2, "fn": _corrupted_state},
    "wrong_tool_arg":           {"step_idx": 3, "fn": _wrong_tool_arg},
    "hallucinated_tool_output": {"step_idx": 3, "fn": _hallucinated_tool_output},
}

def make_fault(name):
    return {**FAULTS[name], "type": name}