from repos import fetch,OUT
import json
r=fetch(('knowno','google-research/google-research'));print(r)
p=OUT/'knowno/source.json';d=json.loads(p.read_text());d['paths']=[x for x in d.get('paths',[]) if x.startswith('language_model_uncertainty/')];p.write_text(json.dumps(d,indent=2));print(d.get('paths'))
