"""One resumable stage of Step 7; all inputs and sources must match the lock."""
import argparse
import fcntl
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scorer_features.common import CACHE,native_flags
from scorer_features.cache import policy_job,vision_job,prediction_job

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['policy','vision','predictions'])
    ap.add_argument('--rank',type=int,default=0);ap.add_argument('--workers',type=int,default=2)
    args=ap.parse_args();native_flags();CACHE.mkdir(parents=True,exist_ok=True)
    lock=(CACHE/f'{args.stage}_{args.rank}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.stage=='policy':policy_job(args.rank,args.workers)
    elif args.stage=='vision':vision_job()
    else:prediction_job()

if __name__=='__main__':main()
