"""Partition the official R2R-CE/RxR-CE training scenes into groups A/B/C; E is val_unseen.

Partitioning is by Matterport scene ID, so an environment shared by R2R-CE and RxR-CE
lands in exactly one group. Scenes are shuffled with a fixed seed and assigned in that
order until each group reaches its target share of the selected training episodes.

    python -I tools/make_scene_partition.py --out manifest/scene_partition.json
"""
import argparse
import collections
import gzip
import json
import random

DATA = "/data/team_a/dave-workspace/RoleVLN/data/datasets"
R2R = DATA + "/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz"
RXR = DATA + "/RxR_VLNCE_v0/{split}/{split}_{role}.json.gz"
GROUP_USE = {
    "A": "policy SFT, DAgger, world-model fitting",
    "B": "scorer branch labels and scorer optimisation",
    "C": "development: checkpoint, setting and override-margin selection",
    "E": "final evaluation only (official val_unseen)",
}


def load_episodes(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)["episodes"]


def scene(ep):
    return ep["scene_id"].split("/")[1]


def selected(split, rxr_roles, rxr_langs):
    eps = {"r2r": load_episodes(R2R.format(split=split)), "rxr": []}
    for role in rxr_roles:
        eps["rxr"] += [e for e in load_episodes(RXR.format(split=split, role=role))
                       if e["instruction"]["language"] in rxr_langs]
    return eps


def counts(eps, scenes):
    return {name: sum(scene(e) in scenes for e in group) for name, group in eps.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fractions", default="0.70,0.15,0.15", help="A,B,C episode shares")
    ap.add_argument("--rxr-roles", default="guide")
    ap.add_argument("--rxr-langs", default="en-US,en-IN")
    args = ap.parse_args()
    roles, langs = args.rxr_roles.split(","), args.rxr_langs.split(",")
    fracs = dict(zip("ABC", map(float, args.fractions.split(","))))

    train = selected("train", roles, langs)
    per_scene = collections.Counter(scene(e) for group in train.values() for e in group)
    order = sorted(per_scene)
    random.Random(args.seed).shuffle(order)

    total = sum(per_scene.values())
    groups, done, bound = {"A": [], "B": [], "C": []}, 0, 0.0
    names = list("ABC")
    for g in names:
        bound += fracs[g] * total
        while order and (g == names[-1] or done + per_scene[order[0]] / 2 < bound):
            s = order.pop(0)
            groups[g].append(s)
            done += per_scene[s]
    val_unseen = selected("val_unseen", roles, langs)
    groups["E"] = sorted({scene(e) for group in val_unseen.values() for e in group})
    assert not set(groups["E"]) & set(per_scene), "val_unseen scene appears in train"

    report = {}
    for g, scenes in groups.items():
        eps = val_unseen if g == "E" else train
        report[g] = {"use": GROUP_USE[g], "num_scenes": len(scenes),
                     "episodes": counts(eps, set(scenes)), "scenes": sorted(scenes)}
    out = {
        "status": "adopted",
        "note": "baseline-v0 pilot split (no earlier pilot split exists); pilot uses less data than a full benchmark.",
        "seed": args.seed,
        "target_episode_fractions": fracs,
        "episode_sources": {
            "r2r_ce": "R2R_VLNCE_v1-3_preprocessed {split}.json.gz (all episodes)",
            "rxr_ce": f"RxR_VLNCE_v0 {{split}}_{{role}}.json.gz, roles={roles}, languages={langs}",
        },
        "unused": "R2R-CE/RxR-CE val_seen (its scenes are training scenes), test splits, EnvDrop",
        "groups": report,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    for g, r in report.items():
        print(g, r["num_scenes"], "scenes", r["episodes"])


if __name__ == "__main__":
    main()
