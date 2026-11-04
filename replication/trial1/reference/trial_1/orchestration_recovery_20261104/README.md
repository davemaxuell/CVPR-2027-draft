# Trial 1 orchestration recovery — 4 November 2026

F training completed at 11:26 KST (857 updates); its checkpoint is verified. The full val-seen evaluation completed all 778 episodes (SR 56.0411%, SPL 51.1014%).

The frozen stage helper checked every spawned command against its preparation directory. Evaluation scripts live under separate run directories, so identity capture failed while evaluation subprocesses continued. The existing supervisor retained their sessions but eventually blocked the pipeline after the repeated error. Original failure logs and queue attempts are preserved in before/ and the shared event/failure ledgers.

launch.py wraps the original sequential stage helper and checks the actual absolute Python script path. It retains stage locking, authorization, adoption of live children, completed-output reuse and bounded retries. The same repair covers F and the queued E/G/J suite, including the separate J collector path. Five CPU subprocess integration tests passed. All 51 F and 141 suite frozen files are unchanged; no training, dataset, checkpoint or evaluation method changed. The wrapper verifies its own frozen source/check receipt before importing the original pipeline.

Only the existing supervisor was gracefully reloaded; the MMC4 process and all unrelated holds were retained. F resumes using its remaining third queue attempt, skipping complete training, checkpoint verification and val-seen; val-unseen is next, then the F comparison and E → G → J. No completed stage is repeated. Trial 2 training remains held.

The copied evaluator uses a generic expanded_stage_two model_role label; this run is Trial 1 F, identified by its dedicated evaluation root, checkpoint hash and continuation receipts. No results are reassigned to Trial 2.
