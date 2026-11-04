# Source and dependency notices

Original project files are copied with source provenance in `SOURCE_ORIGINS.json`. This handoff does not apply a new blanket license to the project's research code or grant rights to datasets/model weights.

`evaluation/vendor/vlnce/measures.py` is the pinned upstream VLN-CE metric source. Its MIT license and source identity are included beside it as `LICENSE` and `SOURCE.json`; retain them with any copy into evaluation outputs.

Other runtime dependencies are installed separately rather than redistributing the server's binary packages. In particular, the reference package metadata identifies `dtw==1.4.0` as GPL-3.0 and `fastdtw==0.3.4` as MIT; their upstream packages supply their licenses. Habitat, PyTorch, Transformers and other dependencies retain their own licenses. StreamVLN-derived protocol references and upstream commit identifiers remain in source/docstrings and the recipe document.

Matterport imagery/geometry, rendered frames, annotation releases, model weights, optimizer states, credentials and live process state are not included in this package.
