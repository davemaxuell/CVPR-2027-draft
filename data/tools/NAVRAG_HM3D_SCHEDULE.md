# Scheduled NavRAG-CE HM3D download

Booked once for **2026-10-23 05:00 Asia/Seoul (KST)** using the user's crontab and
the active system cron daemon. This does not depend on an interactive login session.
The runner rejects execution before that time and on any other date, removes its
own cron entry when it starts, and locks against concurrent execution.

Credentials must be configured before the scheduled time. Edit the private file:

```bash
nano /home/bufsgpu/.config/rolevln/hm3d.netrc
```

Replace the two placeholder values, preserving this format:

```text
machine api.matterport.com
login YOUR_MATTERPORT_TOKEN_ID
password YOUR_MATTERPORT_TOKEN_SECRET
```

Use the API token ID and API token secret associated with approved HM3D access,
not the Matterport account email/password. Obtain them through the
[official Habitat instructions](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#habitat-matterport-3d-research-dataset-hm3d).
The file is mode 0600. Do not add real values to this documentation or chat.
The runner ignores the template placeholders and refuses a credential file accessible
to other users. Credentials are passed to curl on stdin and are not logged.

Check local readiness without starting a transfer:

```bash
/usr/bin/python3 /data/team_a/dave-workspace/RoleVLN/data/tools/download_navrag_hm3d.py --preflight
```

`credentials_available: true` means a private credential file contains non-placeholder
values; it does not by itself confirm that Matterport has approved those credentials.
If credentials are absent at 05:00, the job records `blocked_missing_credentials`
and exits without transferring data.

The job downloads the official HM3D v0.2 Habitat training archive (listed as about
27 GB) and its small configuration archive. Partial downloads are retained for
resumption. It extracts the mesh/navmesh files needed by NavRAG-CE's 612 training
scenes under `data/scene_datasets/hm3d/train/`, preserves existing conflicting files,
and runs the CE data audit before updating the inventory. Original archives and
local SHA-256 receipts remain under `data/datasets/.downloads/HM3D-v0.2/`.

Files relative to the RoleVLN root:

- Schedule: `data/splits/navrag_hm3d_download_schedule.json`
- Job status: `data/splits/navrag_hm3d_download_status.json`
- Log: `data/logs/navrag_hm3d_scheduled_download.log`
- Runner: `data/tools/download_navrag_hm3d.py`
- Inventory: `data/DATASET_INVENTORY.md`
