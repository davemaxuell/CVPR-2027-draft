# HA-VLN 2.0 local dataset

All publicly released dataset files and the CMA checkpoint are downloaded at pinned revisions.

- `HA-R2R/`: train, val_seen and val_unseen, with original and BERT-indexed tokens.
- `HAPS2_0/`: 486 motions, 120 GLB frames per motion; all archive files retained.
- `Multi-Human-Annotations/human_motion.json`: official human placement and motion annotations.
- `HA-R2R-tools/`: vocabulary, embeddings and published collision statistics.
- `checkpoints/HA-VLN-CMA/ckpt.39.pth`: published baseline checkpoint.
- `scene_datasets/`: relative link to the existing shared Matterport3D scenes.
- `original/huggingface/` and `original/github/`: unmodified source files.

Sources: https://github.com/UWMILab/HA-VLN and https://huggingface.co/datasets/fly1113/HA-VLN.
Hugging Face revision: 2632eeac7d8a83570bd9ddbde0116fa1be3d76ff
GitHub revision: 4e1d1fe4ceebee7253d3ba53c04468a8bc2c16c8

Verification: `source_manifest.json`, `haps_extraction_report.json`, and `data/splits/ha_vln_completion_report.json`.

HA-VLN 2.0 mirrors all public train, val_seen and val_unseen episode files in both published tokenizations. No public test episodes are included in this release.
HA-VLN 2.0 includes its published CMA checkpoint. Simulator/Docker installation and training/evaluation execution are outside this dataset download.
HA-VLN-DE currently has only a README placeholder in this source repository; no separate discrete episode release is fabricated.
HA-VLN has 2 published human placements with extra pose entries beyond the simulator's 120-frame cycle. All required frames and referenced assets are valid; the extra entries are preserved unchanged and listed in the completion report.
