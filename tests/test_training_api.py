"""Training-loop helpers: dense masks, truncation, event-log files and replay."""
from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import fastcatan
from fastcatan.events import (EventWriter, ReplayError, check_version, read_events,
                              replay)
from tests.conftest import legal_actions

ACT = fastcatan.action
EV = {name: offset for name, (offset, _) in fastcatan.EVENT_FIELDS.items()}
STATUS = {name: offset for name, (offset, _) in fastcatan.STATUS_FIELDS.items()}


def _random_batch_run(n: int, steps: int, seed: int, path: Path,
                      autoreset: bool = True) -> fastcatan.BatchedEnv:
    """Play `steps` random batched steps, logging everything into `path`."""
    batch = fastcatan.BatchedEnv(n, seed=seed)
    masks = np.zeros((n, fastcatan.MASK_WORDS), dtype=np.uint64)
    actions = np.zeros(n, dtype=np.uint32)
    dones = np.zeros(n, dtype=np.uint8)
    rng = random.Random(seed)
    with EventWriter(path, agents=[f"seat{s}" for s in range(4)], meta={"run": "t"}) as out:
        batch.enable_log(fastcatan.log.LOG_ALL, 4096)
        batch.reset()
        for _ in range(steps):
            batch.write_masks(masks)
            for i in range(n):
                legal = legal_actions(masks[i])
                actions[i] = rng.choice(legal) if legal else fastcatan.NO_ACTION
            (batch.step_autoreset if autoreset else batch.step)(actions, dones)
            out.drain(batch)
            for i in np.flatnonzero(dones):
                if autoreset:
                    out.assign(i, batch.game_id(i), agents=["a", "b", "c", "d"],
                               meta={"slot": int(i)})
    assert batch.log_overflow == 0
    return batch


# --- dense masks -----------------------------------------------------------------

def test_dense_masks_match_packed_masks() -> None:
    n = 16
    batch = fastcatan.BatchedEnv(n, seed=5)
    batch.reset()
    packed = np.zeros((n, fastcatan.MASK_WORDS), dtype=np.uint64)
    dense = np.zeros((n, fastcatan.NUM_ACTIONS), dtype=np.bool_)
    actions = np.zeros(n, dtype=np.uint32)
    dones = np.zeros(n, dtype=np.uint8)
    single = np.zeros(fastcatan.NUM_ACTIONS, dtype=np.bool_)
    rng = random.Random(5)
    for _ in range(300):
        batch.write_masks(packed)
        batch.write_masks_dense(dense)
        for i in range(n):
            legal = legal_actions(packed[i])
            assert np.flatnonzero(dense[i]).tolist() == legal
            actions[i] = rng.choice(legal)
        batch.step_autoreset(actions, dones)

    env = fastcatan.Env()
    env.reset(3)
    for _ in range(300):
        env.action_mask_dense(single)
        assert np.flatnonzero(single).tolist() == env.legal_actions()
        env.step(rng.choice(env.legal_actions()))


def test_dense_mask_buffers_are_checked() -> None:
    batch = fastcatan.BatchedEnv(2)
    batch.reset()
    with pytest.raises(RuntimeError, match="dense mask buffer"):
        batch.write_masks_dense(np.zeros((2, fastcatan.NUM_ACTIONS - 1), dtype=np.bool_))
    with pytest.raises(TypeError):
        batch.write_masks_dense(np.zeros((2, fastcatan.NUM_ACTIONS), dtype=np.uint8))
    env = fastcatan.Env()
    env.reset(0)
    with pytest.raises(RuntimeError, match="action_mask_dense"):
        env.action_mask_dense(np.zeros(3, dtype=np.bool_))


def test_output_buffers_are_never_silently_copied() -> None:
    """A wrong dtype or a non-contiguous view must raise: converting it would
    write into a temporary and leave the caller's array untouched."""
    batch = fastcatan.BatchedEnv(2)
    batch.reset()
    with pytest.raises(TypeError):
        batch.write_status(np.zeros((2, fastcatan.STATUS_INTS), dtype=np.int64))
    with pytest.raises(TypeError):
        batch.write_obs(np.zeros((2, fastcatan.OBS_SIZE), dtype=np.float64))
    with pytest.raises(TypeError):
        batch.write_masks(np.zeros((2, 2 * fastcatan.MASK_WORDS), dtype=np.uint64)[:, ::2])
    with pytest.raises(TypeError):
        batch.step(np.zeros(2, dtype=np.uint32), np.zeros(2, dtype=np.int32))
    with pytest.raises(TypeError):
        batch.drain_events(np.zeros((1, fastcatan.EVENT_INTS), dtype=np.int64))
    env = fastcatan.Env()
    env.reset(0)
    with pytest.raises(TypeError):
        env.write_obs(0, np.zeros(fastcatan.OBS_SIZE, dtype=np.float64))


# --- truncation ------------------------------------------------------------------

def _stall(legal: list[int]) -> int:
    """Never build after setup, so nobody can win and MAX_TURNS must stop it."""
    for preferred in (ACT.ROLL_DICE, ACT.END_TURN):
        if preferred in legal:
            return preferred
    passive = [a for a in legal if a >= ACT.DISCARD_BASE and a < ACT.TRADE_BASE]
    return passive[0] if passive else legal[0]


def test_turn_limit_reports_truncation() -> None:
    env = fastcatan.Env()
    env.reset(11)
    while not env.done:
        assert not env.truncated
        env.step(_stall(env.legal_actions()))
    assert env.winner == fastcatan.NO_PLAYER
    assert env.truncated

    batch = fastcatan.BatchedEnv(1, seed=0)
    batch.load_snapshots(np.frombuffer(env.snapshot(), dtype=np.uint8)[None, :].copy())
    status = np.zeros((1, fastcatan.STATUS_INTS), dtype=np.int32)
    batch.write_status(status)
    assert status[0, STATUS["done"]] == 1
    assert status[0, STATUS["truncated"]] == 1
    assert status[0, STATUS["winner"]] == fastcatan.NO_PLAYER


def test_a_won_game_is_not_truncated() -> None:
    env = fastcatan.Env()
    env.reset(2)
    rng = random.Random(2)
    while not env.done:
        env.step(rng.choice(env.legal_actions()))
    assert env.winner != fastcatan.NO_PLAYER
    assert not env.truncated


# --- event-log files -------------------------------------------------------------

def test_event_file_round_trip_and_replay(tmp_path: Path) -> None:
    path = tmp_path / "run.fcev"
    n = 6
    batch = _random_batch_run(n, 5000, seed=21, path=path, autoreset=False)

    log = read_events(path)
    assert not log.cut
    assert log.header["fastcatan"] == fastcatan.__version__
    check_version(log)
    games = log.games()
    assert [(g.env_idx, g.game_id) for g in games] == [(i, 0) for i in range(n)]
    env = fastcatan.Env()
    for g in games:
        assert g.agents == [f"seat{s}" for s in range(4)]
        assert g.meta == {"run": "t"}
        assert g.seed == batch.game_seed(g.env_idx)
        list(replay(g, env))
        assert env.snapshot() == batch.snapshot(g.env_idx)
        assert g.finished == env.done


def test_autoreset_games_replay_with_their_assignments(tmp_path: Path) -> None:
    path = tmp_path / "run.fcev"
    _random_batch_run(4, 12_000, seed=8, path=path)
    games = read_events(path).games()
    finished = [g for g in games if g.finished]
    assert any(g.game_id > 0 for g in games), "no autoreset happened"
    assert finished
    env = fastcatan.Env()
    for g in games:
        if g.game_id > 0:
            assert g.agents == ["a", "b", "c", "d"]
            assert g.meta == {"run": "t", "slot": g.env_idx}
        list(replay(g, env))
        if g.finished:
            assert env.done and env.winner == g.winner
            assert env.truncated == g.truncated


def test_replay_detects_divergence_and_missing_rows(tmp_path: Path) -> None:
    path = tmp_path / "run.fcev"
    _random_batch_run(1, 400, seed=3, path=path, autoreset=False)
    game = read_events(path).games()[0]
    env = fastcatan.Env()

    tampered = game.rows.copy()
    actions = np.flatnonzero(tampered[:, EV["event_type"]] == fastcatan.log.ACTION)
    tampered[actions[200], EV["handsize"]] += 1
    game_bad = type(game)(game.env_idx, game.game_id, tampered)
    with pytest.raises(ReplayError, match="diverged before step 200"):
        list(replay(game_bad, env))

    gap = np.delete(game.rows, actions[150], axis=0)
    with pytest.raises(ReplayError, match="rows missing before step 151"):
        list(replay(type(game)(game.env_idx, game.game_id, gap), env))

    no_start = game.rows[game.rows[:, EV["event_type"]] != fastcatan.log.GAME_START]
    with pytest.raises(ReplayError, match="no GAME_START"):
        list(replay(type(game)(game.env_idx, game.game_id, no_start), env))


def test_cut_off_file_reads_complete_frames(tmp_path: Path) -> None:
    path = tmp_path / "run.fcev"
    _random_batch_run(2, 300, seed=4, path=path, autoreset=False)
    full = read_events(path)
    path.write_bytes(path.read_bytes()[:-7])
    with pytest.warns(UserWarning, match="cut off"):
        cut = read_events(path)
    assert cut.cut
    assert 0 < len(cut.rows) < len(full.rows)


def test_version_mismatch_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "run.fcev"
    _random_batch_run(1, 10, seed=1, path=path, autoreset=False)
    log = read_events(path)
    log.header["fastcatan"] = "0.0.1"
    with pytest.raises(ReplayError, match="recorded with fastcatan 0.0.1"):
        check_version(log)


def test_writer_validates_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="agents needs 4"):
        EventWriter(tmp_path / "x.fcev", agents=["a"])
    with EventWriter(tmp_path / "x.fcev") as out:
        with pytest.raises(ValueError, match="rows must be shaped"):
            out.write(np.zeros((2, 3), dtype=np.int32))
    with pytest.raises(ValueError, match="not a fastcatan-event-log"):
        (tmp_path / "y.fcev").write_bytes(b"nope")
        read_events(tmp_path / "y.fcev")


# --- conversion to the viewer format ---------------------------------------------

def test_converter_writes_viewer_logs(tmp_path: Path) -> None:
    path = tmp_path / "run.fcev"
    out = tmp_path / "run.jsonl.gz"
    _random_batch_run(3, 6000, seed=17, path=path)
    viz = pytest.importorskip("fastcatan_viz", reason="pip install ./viz")
    subprocess.run([sys.executable, "-m", "fastcatan_viz.from_events", str(path),
                    "-o", str(out), "--finished"], check=True, capture_output=True)
    read_game_log = viz.read_game_log
    logged = read_game_log(out)
    finished = [g for g in read_events(path).games() if g.finished]
    assert len(logged) == len(finished) > 0
    for game, src in zip(logged, finished):
        assert game["header"]["seed"] == str(src.seed)
        assert game["header"]["meta"]["source"]["game_id"] == src.game_id
        assert game["end"]["winner"] == src.winner
        assert game["end"]["truncated"] == src.truncated
        assert len(game["steps"]) == len(src.actions)
