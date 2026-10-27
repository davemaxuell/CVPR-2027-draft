from repos import fetch
from concurrent.futures import ThreadPoolExecutor
jobs=[('fgr2r','YicongHong/Fine-Grained-R2R'),('uln','weixi-feng/ULN'),('landmark_rxr','hekj/Landmark-RxR')]
with ThreadPoolExecutor(max_workers=3) as ex:
 for r in ex.map(fetch,jobs): print(r,flush=True)
