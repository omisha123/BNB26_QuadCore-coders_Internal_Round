"""Tiny QA agent: plan -> retrieve -> reason -> tool -> answer. Every step goes through run_step(),
which does caching (replay-ready) + logging. A step is a pure function of (type, input, state_before).
World: 60 companies, 30 cities, 6 countries. Two task types: hq_population, hq_country."""
import json, random, re, time, uuid, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schema import conn, h
from llm import llm

# ---------- fake world (seeded, so everyone gets the identical world) ----------
_rng = random.Random(42)
_SYL = ["ar", "bel", "cor", "dun", "el", "fen", "gar", "hol", "ish", "jun", "kal", "lor", "mer",
        "nor", "ov", "pel", "qui", "ros", "sil", "tan", "ul", "ver", "wen", "xan", "yor", "zel"]

def _names(n, suffix=""):
    out = set()
    while len(out) < n:
        out.add("".join(_rng.choice(_SYL) for _ in range(2)).capitalize() + suffix)
    return sorted(out)

COUNTRIES = ["Zandria", "Molvia", "Kestria", "Taloria", "Quenland", "Brevia"]
_city_names = _names(30)
_pops = _rng.sample(range(20, 990), 30)               # unique populations
CITY_INFO = {c: {"population": _pops[i] * 1000, "country": COUNTRIES[i % 6]}
             for i, c in enumerate(_city_names)}
CITIES = list(CITY_INFO)
COMPANIES = {co: _rng.choice(CITIES) for co in _names(60, "Corp")}
DOCS = ([f"{co} is headquartered in {city}." for co, city in COMPANIES.items()] +
        [f"{co} was founded by a team of engineers and sells software." for co in COMPANIES] +
        [f"{city} is a city known for its harbor and markets." for city in CITIES])
TASK_TYPES = ["hq_population", "hq_country"]

def make_task(company, task_type="hq_population"):
    city = COMPANIES[company]
    if task_type == "hq_population":
        q, exp = f"What is the population of the city where {company} is headquartered?", \
                 str(CITY_INFO[city]["population"])
    else:
        q, exp = f"In which country is the city where {company} is headquartered?", \
                 CITY_INFO[city]["country"]
    return {"task_id": f"{task_type}_{company}", "task_type": task_type, "question": q, "expected": exp}

def all_tasks():
    return [make_task(co, t) for co in COMPANIES for t in TASK_TYPES]

# ---------- step functions (rule-based mock inside llm()) ----------
def f_plan(inp):
    q = inp["question"]
    company = llm(f"Extract the company name from this question. Reply with ONLY the name, nothing else.\nQuestion: {q}", lambda: re.search(r"(\w+Corp)", q).group(1))
    return {"company": company, "attr": "country" if "country" in q else "population"}

def f_retrieve(inp):
    words = {inp["company"].lower()}
    scored = sorted(DOCS, key=lambda d: -len(words & set(d.lower().replace('.', '').split())))
    return {"docs": scored[:3]}

def f_reason(inp):
    docs = inp["docs"]
    def mock():
        m = re.search(r"headquartered in (\w+)", " ".join(docs))
        return m.group(1) if m else "UNKNOWN"
    city = llm(f"Docs: {docs}\nWhich city is {inp['company']} headquartered in? Reply with ONLY the city name, no punctuation, nothing else.", mock)
    return {"city": city}

def f_tool(inp):
    info = CITY_INFO.get(inp["city"])
    return {"value": info[inp["attr"]] if info else "NOT_FOUND"}   # real tool, no LLM

def f_answer(inp):
    return {"answer": llm(f"The value is {inp['value']}. Reply with ONLY that value, exactly as written, nothing else.",
                          lambda: str(inp["value"]))}

# ---------- executor ----------
class Ctx:
    def __init__(self, run_id, c, fault=None):
        self.run_id, self.c, self.fault, self.state, self.idx = run_id, c, fault, {}, 0

def run_step(ctx, step_type, inp, fn):
    before = json.loads(json.dumps(ctx.state))
    key = h(step_type, inp, before)
    row = ctx.c.execute("SELECT output FROM cache WHERE input_hash=?", (key,)).fetchone()
    t0, err = time.time(), None
    if row:
        out = json.loads(row[0])
    else:
        try: out = fn(inp)
        except Exception as e: out, err = {}, str(e)
        if err is None:                      # never cache a failed call
            ctx.c.execute("INSERT OR REPLACE INTO cache VALUES(?,?)", (key, json.dumps(out)))
    if ctx.fault and ctx.fault["step_idx"] == ctx.idx:      # fault injection: corrupt output AFTER cache
        out = ctx.fault["fn"](out)
    ctx.state[step_type] = out
    ctx.c.execute("INSERT INTO steps VALUES(?,?,?,?,?,?,?,?,?,?)",
        (ctx.run_id, ctx.idx, step_type, json.dumps(inp), json.dumps(out), json.dumps(before),
         json.dumps(ctx.state), key, int((time.time() - t0) * 1000), err))
    ctx.idx += 1
    return out

def run_agent(task, fault=None, c=None):
    c = c or conn()
    run_id = uuid.uuid4().hex[:10]
    ctx = Ctx(run_id, c, fault)
    p = run_step(ctx, "plan", {"question": task["question"]}, f_plan)
    r = run_step(ctx, "retrieve", {"company": p.get("company", "")}, f_retrieve)
    s = run_step(ctx, "reason", {"company": p.get("company", ""), "docs": r.get("docs", [])}, f_reason)
    t = run_step(ctx, "tool", {"city": s.get("city", ""), "attr": p.get("attr", "population")}, f_tool)
    a = run_step(ctx, "answer", {"value": t.get("value")}, f_answer)
    ans = a.get("answer", "")
    outcome = "pass" if ans == task["expected"] else "fail"
    c.execute("INSERT INTO runs VALUES(?,?,?,?,?,?,?)", (run_id, task["task_id"], task["task_type"],
              task["question"], ans, task["expected"], outcome))
    c.execute("INSERT INTO labels VALUES(?,?,?,?)", (run_id, fault["step_idx"] if fault else None,
              fault["type"] if fault else None, int(bool(fault))))
    c.commit()
    return run_id, outcome

if __name__ == "__main__":   # quick smoke test; use agent/generate.py for real data
    c = conn()
    for t in all_tasks()[:3]: print(run_agent(t, c=c))