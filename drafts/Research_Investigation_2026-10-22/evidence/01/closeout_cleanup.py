from pathlib import Path
import json
import re

root = Path(__file__).resolve().parents[2]
path = root / 'evidence/01/notes.json'
notes = json.loads(path.read_text(encoding='utf-8'))

replacements = {
    'assignment01': 'assignment 01',
    'HAMT-e 2e': 'HAMT-e2e',
    'Seq 2Seq': 'Seq2Seq',
    'EnvDrop 75? precisely SR76.59/SPL67.74': 'EnvDrop SR 76.59/SPL 67.74',
    'LR5e-5': 'learning rate 5e-5',
    'LR1e-5': 'learning rate 1e-5',
    'LR2e−5': 'learning rate 2e−5',
    'LR5e−6': 'learning rate 5e−6',
    'R2R6k': 'R2R 6k',
    'reports3': 'reports 3',
    'MLM/SAP/OG100k': 'MLM/SAP/OG for 100k',
    'or−∞': 'or −∞',
    'PDFp21': 'PDF p 21',
    'assignment02': 'assignment 02',
    'cells,15.2G': 'cells, 15.2G',
    'R2R300k': 'R2R 300k',
    'REVERIE200k': 'REVERIE 200k',
    'listings,1.4M': 'listings, 1.4M',
    '5:36 panoramic': '5: 36 panoramic',
    '8V100,20': '8 V100 GPUs, 20',
    'AdamW5e−5': 'AdamW at 5e−5',
    'd 0': 'd_0',
    's 0': 's_0',
    'STOP0': 'STOP 0',
    'code156': 'code lines 156',
    '194,400RGBD': '194,400 RGB-D',
    'and>5m': 'and longer than 5m',
    'local9.6m': 'local 9.6m',
    'lose3D': 'lose 3D',
    '2V100/3 days': '2 V100 GPUs over 3 days',
    '5RxR/SOON': '5 for RxR/SOON',
    'within±10m': 'within ±10m',
    'includesR2R-CE': 'includes R2R-CE',
    'withoutVGGT': 'without VGGT ',
    'withoutSRDF': 'without SRDF',
    'for8actions': 'for 8 actions',
    'has≤3': 'has at most 3',
    'withoutsubtasks': 'without subtasks',
    'withoutstructuredCoT': 'without structured CoT ',
    'and500/300episode': 'and 500/300-episode',
    '3–4t-nDTW': '3–4 points in t-nDTW',
    'Bill 1235813': 'Bill1235813',
    'and§4maps': 'and §4 maps',
    'when>1m': 'when farther than 1m',
    'within.25m': 'within 0.25m',
    'TypeIII56': 'Type III 56',
    '491Gibson': '491 Gibson',
    '5.14Miterations': '5.14M iterations',
    'graph F1.70': 'graph F1 of .70',
    'graphF1=.70': 'graph F1=.70',
    '10 ksteps': '10k steps',
    '8A100≈80h': '8 A100 GPUs for about 80 hours',
    'includesCVDN': 'includes CVDN',
    'andLLaVA23k': 'and 23k LLaVA examples',
    'temperature.01': 'temperature .01',
    'forSOON/REVERIE': 'for SOON/REVERIE',
    'tasks/agents/mp 3d_agent.py': 'tasks/agents/mp3d_agent.py',
    'forCVDN': 'for CVDN',
    'held-outEQA': 'held-out EQA',
    'summarization,3DQA': 'summarization, 3D QA',
    'smallR2R/REVERIE': 'small R2R/REVERIE',
    '800HM3D+61MP3D': '800 HM3D + 61 MP3D',
    '11MP3D': '11 MP3D',
    'onNavRAG': 'on NavRAG',
    'achievesNavRAG': 'achieves NavRAG ',
    'versusScaleVLN': 'versus ScaleVLN ',
    'toREVERIE': 'to REVERIE ',
    'versusNavRAG': 'versus NavRAG',
    'testgeneration': 'test generation',
    'andDBSCAN': 'and DBSCAN',
    '17 kaction-enriched': '17k action-enriched',
    'withAirBERT': 'with AirBERT',
    'reproducedNaviLLM': 'reproduced NaviLLM',
    'R2RvalSPL': 'R2R validation SPL ',
    'withactiondata': 'with action data',
    'actiondata': 'action data',
    'Zero-shot-transferR2R': 'Zero-shot transfer on R2R ',
    'GPT-4VMapGPT': 'GPT-4V MapGPT ',
    '7Bbaselines': '7B baselines',
    '%mostly/perfectly': '% mostly/perfectly',
    '%fall': '% fall',
    '§3COLMAP/experts': '§3 COLMAP/experts',
    '§5.4humanratings': '§5.4 human ratings',
    '§4.3candidate': '§4.3 candidate',
    'DUETobjectbranch': 'DUET object branch',
    'NavRAGno-boxnavigation': 'NavRAG navigation without object boxes',
    'KERMknowledge': 'KERM knowledge',
    'GELA languagealignment': 'GELA language alignment',
    '§6Limitations': '§6 Limitations',
}

metric_names = r'(?:RGSPL|RGS|SDTW|nDTW|t-nDTW|SPICE|SPL|SR|OSR|NE|TL|GP|IoU)'
model_names = r'(?:DUET|UAGM|KERM|MRM|SAP|BEVBert|GridMM|AutoVLN|GOAT|REVERIE|SOON|R2R|RxR-CE|NavRAG|ScaleVLN|BCE|MapGPT)'

def clean(value):
    if isinstance(value, str):
        for old, new in replacements.items():
            value = value.replace(old, new)
        # Apply spacing to prose only. Code paths, identifiers, URLs and hashes
        # are not visited by this function.
        value = re.sub(r'(' + metric_names + r')(?=\d)', r'\1 ', value)
        value = re.sub(r'(' + model_names + r')(?=\d)', r'\1 ', value)
        value = re.sub(r'(?<=\d)(' + metric_names + r')\b', r' \1', value)
        value = re.sub(r'\b(Table|Tables|Figure|Eq|Eqs|Alg|Section)(?=\d)', r'\1 ', value)
        value = re.sub(r'\b(pp?)(?=\d)', r'\1 ', value)
        value = re.sub(r'(?<=\d)p(?=\d|\s\d)', ' p ', value)
        value = value.replace(' and§', ' and §')
        return value
    if isinstance(value, list):
        return [clean(x) for x in value]
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    return value

for entry in notes.values():
    for field in ('problem', 'method', 'novelty', 'results', 'implementation', 'bottlenecks'):
        if field in entry:
            entry[field] = clean(entry[field])

notes['r2r']['authors'] = [name.replace('S?nderhauf', 'Sünderhauf') for name in notes['r2r']['authors']]
notes['hsgm']['method'] = notes['hsgm']['method'].replace(
    'Two consecutive STOP predictions are required for final completion.',
    'The paper specifies two consecutive STOP predictions for final completion; the released code retains the first STOP flag across intervening motion, as detailed below.'
)
notes['hsgm']['bottlenecks'][0]['description'] = (
    'Repeated self-reported completion is not independent verification because the same model can repeat a correlated error. '
    'The paper proposes consecutive STOP confirmation, while the inspected release does not reset the first STOP flag after motion.'
)
notes['hsgm']['bottlenecks'][0]['evidence'] = 'Paper §3.4; INSPECTED CODE src/agent/agent.py 1070–1094; deduction about correlated errors'
notes['hsgm']['bottlenecks'][0]['status'] = 'Uncertain because evidence is insufficient'
notes['weakmap']['implementation'] = notes['weakmap']['implementation'].replace(
    'The source shows map attention and classifier output, supporting the stated implementation mapping.',
    'The source shows map attention and bounded waypoint-regression output; the trainer uses waypoint MSE rather than a primitive-action classifier loss.'
)
for entry in notes.values():
    for b in entry.get('bottlenecks', []):
        if b.get('status') == 'Still unresolved in the reviewed literature':
            b['status'] = 'Still unresolved in reviewed evidence; broader frontier audit required'

notes['speakerfollower']['results'] = (
    'PAPER EXPERIMENT: Table 1, PDF p 7, gives validation-unseen SR 19.9 for the baseline, 24.6 after speaker data augmentation, '
    '39.3 with augmentation plus pragmatic inference, and 54.6 for all three components including panoramic actions. '
    'The same candidate-search comparison gives 30.5 SR using follower scores and 39.3 using the speaker, which isolates reranking '
    'more directly than the full system comparison. Table 2, p 8, reports test SR 53.5 for both the clean selected route and the challenge '
    'submission, but trajectory length changes from 11.63m to 1257.38m when physical search traversal is included; OSR changes from '
    '63.9 to 96.0. These are separate resource protocols. VERDICT: augmentation, pragmatic reranking and action abstraction each contribute. '
    'Speaker likelihood is learned compatibility rather than semantic entailment or calibrated completion confidence, and search gains carry substantial movement cost.'
)
notes['speakerfollower']['method'] += (
    '\n\nSection 4.1, PDF pp 6–7, specifies student forcing: sample follower actions and supervise the shortest-path action from the resulting state. '
    'Speaker maximum-likelihood training uses token cross entropy. The follower trains for 50k iterations on generated instructions and then 20k '
    'on human instructions; image ResNet features are fixed. The reported inference mixture uses λ=0.95.'
)
notes['speakerfollower']['pages'] = sorted(set(notes['speakerfollower']['pages']) | {6, 7, 8})

path.write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding='utf-8')
print('Updated closeout prose and precise evidence records.')
