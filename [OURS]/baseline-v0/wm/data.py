"""Shared root/action IDs for DINO (3 frames) and CDiT (4 frames).

Only observed RGB and executed commands are model inputs. Poses and future
displacements are deliberately absent. Collision is returned as target metadata.
"""
import hashlib
import io
import json
import os
import sqlite3
from functools import lru_cache
from pathlib import Path

from PIL import Image


class TransitionDataset:
    def __init__(self, corpus, history_frames=3, subset="matched", expected_policy_identity=None):
        self.root = Path(corpus).resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        assert self.manifest["status"] == "complete"
        assert self.manifest["format"] == "actual_ce_transitions_v1"
        if expected_policy_identity is not None:
            assert self.manifest["selected_policy_identity_sha256"] == expected_policy_identity, "Policy identity mismatch"
        assert history_frames in (3, 4) and subset in ("matched", "dino_early_extra")
        assert subset == "matched" or history_frames == 3, "Early roots cannot be used for the four-frame comparison"
        self.history_frames, self.subset = history_frames, subset
        self.table = "matched_samples" if subset == "matched" else "early_samples"
        self._conn, self._pid = None, None
        self.count = self.manifest["counts"]["matched" if subset == "matched" else "dino_early_extra"]

    def __len__(self):
        return self.count

    def connection(self):
        if self._conn is None or self._pid != os.getpid():
            self._conn = sqlite3.connect(f"file:{self.root / 'catalog.sqlite'}?mode=ro&immutable=1", uri=True)
            self._conn.row_factory = sqlite3.Row
            self._pid = os.getpid()
        return self._conn

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_conn"], state["_pid"] = None, None
        return state

    @lru_cache(maxsize=64)
    def metadata(self, directory, expected):
        raw = (Path(directory) / "meta.json").read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected, "Episode metadata changed"
        return json.loads(raw)

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        conn = self.connection()
        row = conn.execute(f"SELECT t.*, e.source, e.dataset, e.episode_id, e.scene, e.directory, e.meta_sha256, e.encoding "
                           f"FROM {self.table} s JOIN transitions t ON t.id=s.transition_id JOIN episodes e ON e.id=t.episode_pk WHERE s.slot=?", (int(index),)).fetchone()
        meta = self.metadata(row["directory"], row["meta_sha256"])
        t = row["t"]
        first = max(0, t - self.history_frames + 1)
        if self.subset == "matched":
            assert t >= 3 and t - first + 1 == self.history_frames
        def frame(relative):
            expected = conn.execute("SELECT sha256 FROM frames WHERE episode_pk=? AND relative_path=?", (row["episode_pk"], relative)).fetchone()[0]
            raw = (Path(row["directory"]) / relative).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == expected, ("Frame changed", relative)
            with Image.open(io.BytesIO(raw)) as image:
                return image.convert("RGB")
        extension = "png" if row["encoding"] == "PNG" else "jpg"
        history = [frame(f"rgb/{i:04d}.{extension}") for i in range(first, t + 1)]
        target = frame(row["target_frame"])
        root_id = f"{row['source']}/{row['dataset']}/{row['episode_id']}/t{t:06d}"
        return {"sample_id": root_id + f"/a{row['action']}", "root_id": root_id,
                "observations": history, "executed_history_actions": meta["executed_action"][first:t],
                "action": row["action"], "movement_action_index": row["action"] - 1,
                "target_observation": target, "collision_target": bool(row["collided"]),
                "source": row["source"], "dataset": row["dataset"], "scene": row["scene"],
                "episode_id": row["episode_id"], "t": t, "is_branch": bool(row["is_branch"])}
