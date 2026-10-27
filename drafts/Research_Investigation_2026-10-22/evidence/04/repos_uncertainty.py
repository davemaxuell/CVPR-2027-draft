from repos import fetch
from concurrent.futures import ThreadPoolExecutor
import requests,re
jobs=[('explore_confident','Stanford-ILIAD/explore-eqa'),('m3e','YongliangJiang/M3E-Code')]
with ThreadPoolExecutor(max_workers=2) as ex:
 for r in ex.map(fetch,jobs): print(r,flush=True)
for url in ['https://robot-help.github.io/','https://explore-eqa.github.io/']:
 h=requests.get(url).text
 print(url,sorted(set(re.findall(r'href=[\"\x27]([^\"\x27]+)',h))))
