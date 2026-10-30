# viz/ — game logs and viewer

Development tooling kept in the repo but not part of the `fastcatan` package:
`pyproject.toml` excludes this folder from the sdist, and the wheel never
contains it. Recorded logs in `viz/logs/` stay local (git-ignored).

Evaluation games are written as **detailed game logs**: one JSON line per step
carrying the action, a readable description, the legal actions the actor had,
optional agent output, and the full state after the step. The viewer draws
straight from that file — no simulator, agent, or checkpoint is involved, so a
log stays viewable after models are deleted or the simulator changes.

Run the Python parts with a Python that has `fastcatan` 2.1+ (the anaconda
`python3` here; the VS Code workspace interpreter `/bin/python3` has none).

To view a game, open `viz/viewer.html` in a browser and drop a `.jsonl` /
`.jsonl.gz` log on it, or use its **Open log…** button (several files at once
are fine). `viz/logs/demo.jsonl.gz` is a sample; make more with

```bash
python3 viz/record_demo.py          # 4 demo games -> viz/logs/demo.jsonl.gz
```

## Logging evaluation games

```python
import sys; sys.path.insert(0, "viz")
from gamelog import GameLogWriter

with GameLogWriter("viz/logs/eval.jsonl.gz") as log:       # any number of games per file
    for seed in seeds:
        env.reset(seed)
        rec = log.start_game(env, seed=seed, agents=["ppo-v3", "random", "random", "random"],
                             meta={"checkpoint": "ckpt_120.zip"})
        while not env.done:
            action = agents[env.actor_to_act].act(env)
            env.step(action)
            rec.record(action)                               # always right after the step
```

- **BatchedEnv**: `log.start_game(batch, env_idx=i, ...)` (the seed is read from
  the slot), then `rec.record(actions[i])` after each `batch.step(...)`. Use
  `step`, not `step_autoreset`, so the slot is not replaced mid-log.
- **Agent output (later)**: `rec.record(action, info={"value": 0.31, "top": [...]})`
  stores a dict with the step; the viewer already shows it under the current
  move. numpy / torch values are converted, and NaN / inf are stored as strings.
- **Killed runs**: each finished game is flushed to disk, so a run that dies
  midway keeps every completed game; the viewer and `read_game_log` load a
  cut-off file up to the cut and say so.
- Cost: about 50 µs per step; a 1,000-step game is roughly 30 KB gzipped.

## Viewer controls

| Key | Action |
|---|---|
| Space | play / pause |
| ← / → | step back / forward |
| Shift + ← / → | previous / next turn |
| Home / End | first / last step |
| + / − | autoplay speed |
| s | skip trade-draft and no-op steps |
| i | node / edge / hex ID overlay |

The view is omniscient: every hand and development card ("new" = bought this
turn), with public VP next to true VP when they differ. Hover the
"1 of N legal" note to see the options the actor had.

## Log format (version 1)

| Record | Fields |
|---|---|
| `game` | `format`, `version`, `fastcatan`, `seed` (string), `agents[4]`, `meta`, `board` (hex resources/numbers, port types), `topology`, `action_names`, `state` |
| `step` | `i`, `actor`, `action`, `name`, `type`, `text`, `legal`, `hl`?, `info`?, `warnings`?, `state` |
| `end` | `winner`, `turns`, `steps`, `vp[4]` (absent if the game did not finish) |

`state` holds turn, phase, flag, current/acting seat, dice, robber, awards, dev
deck size, bank, `nodes` (54 chars: `.`, `a`–`d` settlement, `A`–`D` city by
seat), `edges` (72 chars: `.` or seat), the open trade, and per seat VP, public
VP, resources, dev cards (`dev` playable, `dev_new` bought this turn), knights,
road length, pieces left, cards owed to a discard, and port bits.
`gamelog.read_game_log(path)` loads a file back in Python.

## Files

| File | Role |
|---|---|
| `gamelog.py` | detailed log format: `GameLogWriter`, `read_game_log` |
| `viewer.html` | the viewer (vanilla JS/SVG, no dependencies) |
| `record_demo.py` | demo evaluation games with simple random agents |
