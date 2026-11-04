"""RxR-CE episodes for habitat-lab 0.2.4, registered as dataset type "RxRVLN-v1".

habitat-lab's R2RVLN-v1 loader requires an instruction vocabulary and rejects RxR's extra
instruction fields. This loader reads {split}_{role}.json.gz for each configured role, keeps
only the configured languages, and preserves RxR's instruction metadata on the episode.
"""
import gzip
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

import attr

from habitat.config.default_structured_configs import DatasetConfig
from habitat.core.registry import registry
from habitat.datasets.utils import VocabDict
from habitat.datasets.vln.r2r_vln_dataset import DEFAULT_SCENE_PATH_PREFIX, VLNDatasetV1
from habitat.tasks.nav.nav import NavigationGoal
from habitat.tasks.vln.vln import InstructionData, VLNEpisode


@dataclass
class RxRDatasetConfig(DatasetConfig):
    roles: List[str] = field(default_factory=lambda: ["guide"])
    languages: List[str] = field(default_factory=list)  # empty keeps every language


@attr.s(auto_attribs=True, kw_only=True)
class RxRInstructionData(InstructionData):
    instruction_id: Optional[int] = None
    language: Optional[str] = None
    annotator_id: Optional[str] = None
    edit_distance: Optional[float] = None
    timed_instruction: Optional[list] = None


@registry.register_dataset(name="RxRVLN-v1")
class RxRVLNDatasetV1(VLNDatasetV1):
    @staticmethod
    def check_config_paths_exist(config) -> bool:
        paths = [config.data_path.format(split=config.split, role=role) for role in config.roles]
        return all(os.path.exists(p) for p in paths) and os.path.exists(config.scenes_dir)

    def __init__(self, config=None) -> None:
        self.episodes = []
        self.instruction_vocab = VocabDict(word_list=[])
        if config is None:
            return
        languages = set(config.languages)
        for role in config.roles:
            with gzip.open(config.data_path.format(split=config.split, role=role), "rt") as f:
                for raw in json.load(f)["episodes"]:
                    if not languages or raw["instruction"]["language"] in languages:
                        self.episodes.append(self._episode(raw, config.scenes_dir))
        self.episodes = list(filter(self.build_content_scenes_filter(config), self.episodes))

    @staticmethod
    def _episode(raw, scenes_dir: Optional[str]) -> VLNEpisode:
        episode = VLNEpisode(**raw)
        if scenes_dir is not None:
            if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                episode.scene_id = episode.scene_id[len(DEFAULT_SCENE_PATH_PREFIX):]
            episode.scene_id = os.path.join(scenes_dir, episode.scene_id)
        episode.instruction = RxRInstructionData(**episode.instruction)
        episode.goals = [NavigationGoal(**goal) for goal in episode.goals]
        return episode
