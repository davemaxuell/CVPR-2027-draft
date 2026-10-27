from repos import fetch
from concurrent.futures import ThreadPoolExecutor
import json
jobs=[('aiuta','intelligolabs/CoIN'),('mind_error','intelligolabs/R2RIE-CE'),('ieve','XiaohanLei/IEVE'),('slow4fast','yl6017339/Slow4Fast-VLN'),('iqa','danielgordon10/thor-iqa-cvpr-2018'),('vsi_bench','vision-x-nyu/thinking-in-space')]
with ThreadPoolExecutor(max_workers=4) as ex:
 for r in ex.map(fetch,jobs):print(json.dumps(r),flush=True)
