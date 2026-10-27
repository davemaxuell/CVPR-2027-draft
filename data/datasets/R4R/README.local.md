# R4R local generation

The three JSON splits are complete outputs of the official Google Research
generator, using the original R2R splits, the official Matterport connectivity
graphs, a 3.0 metre joining threshold, and no heading threshold.

| Split | Generated paths | Generated instructions | README example paths | README example instructions |
|---|---:|---:|---:|---:|
| train | 25,921 | 233,532 | 25,930 | 233,613 |
| val_seen | 115 | 1,035 | 115 | 1,035 |
| val_unseen | 5,026 | 45,234 | 5,018 | 45,162 |

The current official recipe does not reproduce the historical counts printed
in the upstream README. NetworkX 2.3 (the documented original version) and
3.4.2 produce the same counts shown here. All 90 local connectivity files
match the official Matterport3DSimulator repository byte for byte.

An independent implementation of Dijkstra's algorithm verifies every eligible
ordered pair, the shortest connector, the composed paths, and all combined
instructions. No eligible pairs are missing. Original annotations,
connectivity, generator code, and the distance threshold were not modified.

`generation_manifest.json` records the pinned upstream commit, source and
output SHA-256 hashes, dependency versions, and composition-audit results.
The upstream generator and graph helper are saved in `upstream/`.

Source: https://github.com/google-research/google-research/tree/master/r4r
