"""Read-only CPU selection audit for the one authorized fresh J collection."""
import sys
from types import SimpleNamespace
from context import *

sys.path.insert(0, str(SOURCE))
from tools.collect_stream_dagger import chosen_episodes, rule_from


def main():
    args = SimpleNamespace(gpu=0, episodes=0, seed=1, rule_overrides='')
    selected = {}
    for dataset in ['r2r','rxr']:
        _, episodes, lengths = chosen_episodes(args, dataset)
        selected[dataset] = [str(ep.episode_id) for ep in episodes]
        assert len(selected[dataset]) == len(set(selected[dataset]))
        assert set(selected[dataset]) == set(lengths)
    assert {k:len(v) for k,v in selected.items()} == {'r2r':10819,'rxr':19408}
    result = dict(status='complete', time=utc(), selected_episodes=selected,
        counts={k:len(v) for k,v in selected.items()}, beta=0, seed=1, stop_bias=0,
        rule=rule_from(args).as_dict(), frozen_policy='F final epoch, bound only after checkpoint verification',
        note='Collector follows the existing validated-replay rule, including six RxR episodes too short for expert SFT. Corrective training again applies the minimum-four-action rule.')
    atomic_json(HERE/'collection/plan.json', result)
    print(result['counts'])


if __name__ == '__main__':main()
