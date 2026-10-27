# GSA-R2R local dataset

Official source: https://github.com/honghd16/GSA-VLN

`R2R/annotations/ESA_Dataset` contains all 28 official JSON files, preserving
the upstream Train, Validation and Test directory structure. `R2R/connectivity`
and `R2R/connectivity_mp3d` come from the official ScaleVLN preprocessing archive.
`R2R/features/clip_vit-b16_mp3d_hm3d_gibson.hdf5` is the exact CLIP ViT-B/16
member required by the GSA-R2R baseline. Other ScaleVLN preprocessing files
are retained alongside these annotations.

Set the upstream `--root_dir` to this GSA-R2R directory. The upstream parser
also hardcodes relative feature/connectivity paths: point `img_ft_file` and
`connectivity_dir` to the files above when running from another code checkout.
The baseline uses MatterSim with rendering disabled and these feature/graph
assets. Raw HM3D meshes are not included in this release mirror.

The feature member is downloaded using verified HTTP byte ranges, decompressed
and checked against its upstream ZIP CRC. Its local SHA-256 and range receipts
are recorded under `data/splits` and `data/datasets/.downloads`.
The full 37.5 GB multi-model feature archive is not needed for this baseline.
Model checkpoints are optional; `pano_inputs.h5` is a regenerable pretraining
cache (see the upstream README), rather than an additional annotation release.

Validation results: `data/splits/gsa_r2r_completion_report.json`.
