# fastcatan-viz — game logs and viewer

Game logs, event-log replay, and a browser viewer for
[fastcatan](https://github.com/s1nbo/fastcatan). It is a separate package so
the simulator carries none of it, and it lives only in this repository (not on
PyPI):

```bash
pip install "fastcatan-viz @ git+https://github.com/s1nbo/fastcatan#subdirectory=viz"
pip install ./viz              # from a checkout
```

It has no dependencies beyond `fastcatan` and the standard library, and is
released together with it. A log replays only on the fastcatan version that
recorded it, so `fastcatan-viz` pins the same minor version.

There are two ways to get a game into the viewer:

| Source | Contains | Use for |
|---|---|---|
| `fastcatan.events.EventWriter` file (`.fcev`) | seed + actions per game, compact | training runs: log everything, look at a few games later |
| `GameLogWriter` file (`.jsonl.gz`) | full state per step, plus optional agent output | evaluation games, where you want values / policies shown |

The viewer reads only `.jsonl` / `.jsonl.gz`. Convert an event-log file first;
the converter replays the games through the simulator. No agents or
checkpoints are involved either way, so a log stays viewable after the models
are gone.

## Converting training logs

```bash
fastcatan-viz-convert run.fcev --list                       # games in the file
fastcatan-viz-convert run.fcev --finished --limit 20        # -> run.jsonl.gz
fastcatan-viz-convert run.fcev --env 3 --game 7 -o one.jsonl.gz
```

The converter refuses a file recorded with another fastcatan version (pass
`--allow-version-mismatch` to try anyway). Every step is checked against the
recorded rows. A game that diverges, or lost rows to a log overflow, is
skipped with a message instead of being written halfway. Agents and metadata
assigned with `EventWriter(agents=..., meta=...)` or `assign(...)` appear in the
viewer, together with a `source` entry pointing back to the file, slot, and
game.

## Logging evaluation games

```python
from fastcatan_viz import GameLogWriter

with GameLogWriter("logs/eval.jsonl.gz") as log:          # any number of games per file
    for seed in seeds:
        env.reset(seed)
        rec = log.start_game(env, seed=seed, agents=["ppo-v3", "random", "random", "random"],
                             meta={"checkpoint": "ckpt_120.zip"})
        while not env.done:
            action = agents[env.actor_to_act].act(env)
            env.step(action)
            rec.record(action, info={"value": 0.31})         # always right after the step
```

- **BatchedEnv**: `log.start_game(batch, env_idx=i, ...)` (the seed is read from
  the slot), then `rec.record(actions[i])` after each `batch.step(...)`. Use
  `step`, not `step_autoreset`, so the slot is not replaced mid-log.
- **Agent output**: `info` is a free-form dict shown under the current move.
  NumPy and torch values are converted, and NaN / inf are stored as strings.
- **Killed runs**: each finished game is flushed to disk, so a run that dies
  midway keeps every completed game; the viewer and `read_game_log` load a
  cut-off file up to the cut and say so.
- Cost: about 50 µs per step; a 1,000-step game is roughly 30 KB gzipped.

`fastcatan-viz-demo` records 4 demo games with simple random agents into
`logs/demo.jsonl.gz`.

## Viewer

`fastcatan-viz` opens the bundled `viewer.html` in a browser (it also prints
the path; `fastcatan_viz.viewer_path()` returns it). Drop one or more
`.jsonl` / `.jsonl.gz` logs on it, or use **Open log…**. It is a single
vanilla JS/SVG file that never touches the network.

| Key | Action |
|---|---|
| Space | play / pause |
| ← / → | step back / forward |
| Shift + ← / → | previous / next turn |
| Home / End | first / last step |
| + / − | autoplay speed |
| s | cycle the trade filter: all steps · hide drafting · successful trades only |
| i | node / edge / hex ID overlay |

The view is omniscient: every hand and development card ("new" = bought this
turn), with public VP next to true VP when they differ. Hover the
"1 of N legal" note to see the options the actor had. A game stopped by the
turn limit is labelled truncated.

The **Trades** filter hides trade negotiation. *Hide drafting* drops the
offer-composing steps and illegal no-ops; *Successful only* also drops opening,
accepting, declining, and cancelling proposals, keeping just the trades that
went through. Hidden steps leave the event log and are skipped when stepping
or playing; the choice is remembered.

Below the board, a victory-point timeline shows each seat's true VP over the
whole game; hover it for the scores at any step and click to jump there. The
theme button cycles between following the system, light, and dark.

### Figures for papers

**Export** writes the board or the VP timeline at the current step as a
standalone SVG (vector, for LaTeX via `\includesvg` or after conversion to PDF)
or a 4× PNG. Figures always use the light print palette, whatever the page
theme, and carry no page styling. Each resource tile also has a glyph, so the
board stays readable in grayscale print. Options in the menu drop the
current-move highlight from the board or the step cursor from the timeline.
Files are named `catan_game<N>_step<S>_board.svg` and so on.

Player colours (red, blue, orange, magenta) were checked for colour-vision
deficiency separation in both themes, and every player name is printed next to
its colour, never in colour alone.

## Game-log format (version 1)

| Record | Fields |
|---|---|
| `game` | `format`, `version`, `fastcatan`, `seed` (string), `agents[4]`, `meta`, `board` (hex resources/numbers, port types), `topology`, `action_names`, `state` |
| `step` | `i`, `actor`, `action`, `name`, `type`, `text`, `legal`, `hl`?, `info`?, `warnings`?, `state` |
| `end` | `winner`, `turns`, `steps`, `vp[4]`, `truncated` (absent if the game did not finish) |

`state` holds turn, phase, flag, current/acting seat, dice, robber, awards, dev
deck size, bank, `nodes` (54 chars: `.`, `a`–`d` settlement, `A`–`D` city by
seat), `edges` (72 chars: `.` or seat), the open trade, and per seat VP, public
VP, resources, dev cards (`dev` playable, `dev_new` bought this turn), knights,
road length, pieces left, cards owed to a discard, and port bits.
`fastcatan_viz.read_game_log(path)` loads a file back in Python.

## Files

| File | Role |
|---|---|
| `fastcatan_viz/gamelog.py` | detailed log format: `GameLogWriter`, `read_game_log` |
| `fastcatan_viz/from_events.py` | `fastcatan-viz-convert`: event-log file → game log |
| `fastcatan_viz/record_demo.py` | `fastcatan-viz-demo`: demo games with random agents |
| `fastcatan_viz/viewer.html` | the viewer |
