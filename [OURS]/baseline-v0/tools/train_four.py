import argparse
import fcntl
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0'); os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from scorer_four.training_common import RUN
from scorer_four.train import train


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--variant', choices=['pooled','null'], required=True)
    args = parser.parse_args(); folder = RUN / args.variant; folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / 'train.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    train(args.variant)
