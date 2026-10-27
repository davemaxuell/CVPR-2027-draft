DATA_ROOT=/egr/research-hlr2/zhan1624/mapgpt/
outdir=exprs_map/test/

flag="--root_dir ${DATA_ROOT}
      --img_root /egr/research-hlr2/zhan1624/mapgpt/observations
      --split MapGPT_72_scenes_processed
      --end 3  # the number of cases to be tested
      --output_dir ${outdir}
      --max_action_len 10
      --save_pred
      --stop_after 3
      --llm gpt-4o-2024-05-13
      --response_format json
      --max_tokens 1000
      "

python /localscratch/zhan1624/MapGPT/main_gpt.py $flag
