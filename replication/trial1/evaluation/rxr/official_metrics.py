"""Load verbatim official VLN-CE metric implementations on Habitat 0.2.4.

Only the metric definitions are loaded, so the older upstream visualization/task imports
do not replace the installed Habitat runtime. SOURCE.json pins the unmodified source.
"""
import ast
import gzip
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Union

sys.path.insert(0, str(Path(__file__).resolve().parent / 'dependencies/sim'))

import numpy as np
from dtw import dtw
from fastdtw import fastdtw
from habitat.core.embodied_task import EmbodiedTask, Measure
from habitat.core.registry import registry
from habitat.core.simulator import Simulator
from habitat.tasks.nav.nav import DistanceToGoal, Success
from habitat.config.default_structured_configs import MeasurementConfig
from omegaconf import DictConfig

from common import OUTPUT, sha256


def load_classes():
    source = OUTPUT / 'vendor/vlnce/measures.py'
    provenance = json.loads((source.parent / 'SOURCE.json').read_text())
    if sha256(source) != provenance['sha256']:
        raise RuntimeError('Official metric source changed.')
    names = {'euclidean_distance', 'PathLength', 'OracleNavigationError', 'OracleSuccess',
             'OracleSPL', 'StepsTaken', 'NDTW', 'SDTW'}
    parsed = ast.parse(source.read_text())
    nodes = [n for n in parsed.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    if {n.name for n in nodes} != names:
        raise RuntimeError('Official metric source no longer has the expected definitions.')
    class RxRRoles:
        annotation_roles = ['guide', 'follower']
    namespace = dict(Any=Any, List=List, Union=Union, np=np, ndarray=np.ndarray,
                     gzip=gzip, json=json, dtw=dtw, fastdtw=fastdtw, Config=DictConfig,
                     Measure=Measure, Simulator=Simulator, EmbodiedTask=EmbodiedTask,
                     DistanceToGoal=DistanceToGoal, Success=Success, registry=registry,
                     RxRVLNCEDatasetV1=RxRRoles)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    return {name: namespace[name] for name in names}


CLASSES = load_classes()


@dataclass
class OfficialMeasureConfig(MeasurementConfig):
    type: str = ''


@dataclass
class OfficialOracleSuccessConfig(OfficialMeasureConfig):
    SUCCESS_DISTANCE: float = 3.0


@dataclass
class OfficialNDTWConfig(OfficialMeasureConfig):
    FDTW: bool = True
    GT_PATH: str = ''
    SPLIT: str = ''
    SUCCESS_DISTANCE: float = 3.0


def configure(cfg, gt_path, split):
    from habitat.config import read_write
    from omegaconf import OmegaConf
    with read_write(cfg):
        for name in ['PathLength', 'OracleNavigationError', 'OracleSuccess', 'OracleSPL', 'StepsTaken', 'NDTW', 'SDTW']:
            if name == 'OracleSuccess':
                values=OfficialOracleSuccessConfig(type=name)
            elif name == 'NDTW':
                values=OfficialNDTWConfig(type=name,GT_PATH=str(gt_path),SPLIT=split)
            else:values=OfficialMeasureConfig(type=name)
            # Habitat re-creates each node during task construction. Plain dict nodes
            # assigned into Dict[str, MeasurementConfig] fail that round trip.
            cfg.habitat.task.measurements[CLASSES[name].cls_uuid] = OmegaConf.structured(values)
    return cfg
