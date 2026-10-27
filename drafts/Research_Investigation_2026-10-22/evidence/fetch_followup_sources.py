from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,requests,fitz

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'followups'/'sources';DEST.mkdir(exist_ok=True)
sources={
 'cast':('CAST','https://arxiv.org/pdf/2508.13446v2'),
 'retrospective_rectification':('Nipping the Drift in the Bud','https://arxiv.org/pdf/2602.06356'),
 'realm':('From Region Arrival to Instance-Level Grounding','https://arxiv.org/pdf/2607.03792'),
 'janusvln':('JanusVLN','https://arxiv.org/pdf/2509.22548'),
 'worldmap':('WorldMAP','https://arxiv.org/pdf/2604.07957'),
 'diagnosing_vln':('Diagnosing Vision-and-Language Navigation','https://aclanthology.org/2022.naacl-main.438.pdf'),
 'aps':('Counterfactual VLN via Adversarial Path Sampler','https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510069.pdf'),
 'counterfactual_unseen':('Counterfactual VLN Unravelling the Unseen','https://proceedings.neurips.cc/paper/2020/file/39016cfe079db1bfb359ca72fcba3fd8-Paper.pdf'),
 'instruction_evaluation':('On the Evaluation of VLN Instructions','https://arxiv.org/pdf/2101.10504'),
 'pijepa':('Policy-Guided World Model Planning','https://openaccess.thecvf.com/content/CVPR2026W/WDFM-EAI/papers/Chahe_Policy-Guided_World_Model_Planning_for_Language-Conditioned_Visual_Navigation_CVPRW_2026_paper.pdf'),
 'humanoidvln':('HumanoidVLN','https://arxiv.org/pdf/2608.12860'),
 'closer':('CLOSER: Evidence-Aware Active Visual Verification','https://arxiv.org/pdf/2610.28476v1'),
 'vernav':('VerNav: Verifier-First Low-Latency VLN','https://arxiv.org/pdf/2609.00920v1'),
 'cpnav':('CPNAV: Calibrated Vision-Language Navigation','https://practice-lab-ucla.github.io/paper_web/cpnav/IROS_CP_Nav.pdf'),
 'ua_nwm':('Uncertainty-Aware World Model for Aerial Image-Goal Navigation','https://arxiv.org/pdf/2608.05597v1'),
 'mind_gap':('Mind the Gap: Revisiting Oracle Success Routes','https://arxiv.org/pdf/2308.03244v1'),
 'drive_nav':('DRIVE-Nav','https://arxiv.org/pdf/2603.28691'),
 'adaptive_rollout':('Adaptive Rollout Truncation Based on Epistemic Uncertainty','https://arxiv.org/pdf/2609.21482'),
 'rsi_nav':('RSI-Nav: Policy-Synchronized Language-Action Dual Bootstrapping','https://arxiv.org/pdf/2610.15782v1'),
 'core':('CoRe: Inference-Time Counterfactual Realignment','https://arxiv.org/pdf/2608.14822v1'),
}
def fetch(item):
    key,(title,url)=item
    existing=DEST/(key+'.json')
    if existing.exists():
        previous=json.loads(existing.read_text(encoding='utf-8'))
        if previous.get('status')=='Available' and Path(previous.get('pdf','')).is_file():
            print(key,'Cached',flush=True)
            return previous
    record=dict(id=key,title_lead=title,source_url=url,retrieved='2026-10-22',inspection='Downloaded only; substantive review pending')
    try:
        r=requests.get(url,timeout=60);r.raise_for_status()
        assert r.content[:5]==b'%PDF-',r.headers.get('content-type')
        pdf=DEST/(key+'.pdf');pdf.write_bytes(r.content)
        with fitz.open(pdf) as d:
            pages=[p.get_text(sort=True) for p in d]
            txt=DEST/(key+'.txt');txt.write_text('\n\n'.join(f'=== PDF PAGE {n+1} ===\n{t}' for n,t in enumerate(pages)),encoding='utf-8')
        record.update(pdf=str(pdf),text=str(txt),pages=len(pages),sha256=hashlib.sha256(r.content).hexdigest(),status='Available')
    except Exception as e:record.update(status='Unavailable',error=str(e))
    (DEST/(key+'.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(key,record['status'],flush=True)
    return record
with ThreadPoolExecutor(max_workers=5) as pool:results=list(pool.map(fetch,sources.items()))
(DEST/'source_manifest.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
