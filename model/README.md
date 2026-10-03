# Black Box – Person 2 (model + evaluation)

    python -m bb_model.run_all --db blackbox.db --split split.json     # train on `train`, evaluate 3 test splits
    uvicorn bb_model.api:app --port 8001                               # /diagnose  /verify  /metrics

Layout: bb_model/{loader,features,model,evaluate,run_all,api}.py  (+ __init__.py, empty)

## Rules we follow
- Train ONLY on split.json `train`. Report test_seen / test_unseen_fault / test_unseen_task separately.
- The `cache` table is NOT a feature: it stores the clean output of every step, so "output != cache" finds
  the faulty step 100% of the time (a label leak). It belongs to the replay engine.
- `expected` / `outcome` / `final_answer` are never model inputs.
- /diagnose is meant for runs already known to have failed.
