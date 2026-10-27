python3 do_utils/extract_room_type.py \
    --connectivity_dir ../connectivity \
    --scan_dir /root/mount/Matterport3DSimulator/VLN-BEVBert/data/v1/scans/ \
    --output_file ../datasets/R2R/features/pano_roomtypes.tsv \
    --batch_size 2 \
    --num_workers 1 \
    --model_path ../cache/models--Salesforce--blip-vqa-base/snapshots/c7df8e7cd7aa2ee9af18f56e2b29e59a92651b64/