from pathlib import Path
import requests,re,json,concurrent.futures,fitz
ROOT=Path(__file__).resolve().parents[2]
sites={
'auxthink':'https://horizonrobotics.github.io/robot_lab/aux-think',
'progressthink':'https://horizonrobotics.github.io/robot_lab/progress-think',
'idealvln':'https://allenxinn.github.io/IDEALVLN',
'navfom':'https://pku-epic.github.io/NavFoM-Web/',
'ca_nav':'https://chenkehan21.github.io/CA-Nav-project/',
'streamvln':'https://streamvln.github.io/',
'psgnav':'https://psg-nav.github.io/',
'navid':'https://pku-epic.github.io/NaVid/',
'navila':'https://navila-bot.github.io/',
'uninavid':'https://roboticsproceedings.org/rss21/p013.html',
'ecot':'https://embodied-cot.github.io/',
'rl4vla':'https://rlvla.github.io/',
'discussnav':'https://sites.google.com/view/discussnav',
'analogical':'https://github.com/zhangyuejoslin/VLN-Analogical-Reasoning',
}
def fetch(kv):
 k,u=kv
 try:
  r=requests.get(u,timeout=45); r.raise_for_status()
  (ROOT/'evidence'/'03'/(k+'_page.html')).write_text(r.text,encoding='utf-8')
  links=sorted(set(re.findall(r'https?://[^\s<>\"\x27]+',r.text)))
  use=[x for x in links if any(v in x for v in ['github.com/','arxiv.org/','huggingface.co/'])]
  return {'id':k,'url':u,'links':use,'status':r.status_code}
 except Exception as e:return {'id':k,'url':u,'error':str(e)}
out=list(concurrent.futures.ThreadPoolExecutor(6).map(fetch,sites.items()))
(ROOT/'evidence'/'03'/'project_sources.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
for x in out: print(json.dumps(x,ensure_ascii=True))
