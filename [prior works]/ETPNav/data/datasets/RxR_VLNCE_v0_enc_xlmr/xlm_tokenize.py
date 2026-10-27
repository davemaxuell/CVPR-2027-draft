import glob
import gzip
import json
from tqdm import tqdm
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained('xlm-roberta-base', do_lower_case=True)

fns = glob.glob('data/datasets/RxR_VLNCE_v0_enc_xlmr/*/*_raw.json.gz')
for fn in fns:
    with gzip.open(fn) as f:
        data = json.load(f)

    split = fn.split('/')[-2]
    print('Working on %s split ...' % (split) )

    for i, ep in tqdm(enumerate(data['episodes']), total=len(data['episodes']) ):
        instr_dict = ep['instruction']
        instr_dict.pop('timed_instruction') if 'timed_instruction' in instr_dict else None
        tok_output = tok(instr_dict['instruction_text'])
        instr_dict['instruction_tokens'] = tok_output['input_ids']
        data['episodes'][i]['instruction'] = instr_dict

    json_str = json.dumps(data)
    json_bytes = json_str.encode('utf-8')
    with gzip.open('data/datasets/RxR_VLNCE_v0_enc_xlmr/%s/%s_guide.json.gz' % (split, split), 'w') as fout:
        fout.write(json_bytes)