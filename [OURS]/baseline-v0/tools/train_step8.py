import argparse
import fcntl
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scorer_training.common import RUN
from scorer_training.train import train

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('variant',choices=['pooled','null']);a=ap.parse_args()
    out=RUN/a.variant;out.mkdir(parents=True,exist_ok=True)
    lock=(out/'train.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    train(a.variant)
