"""Small independent control-flow checks; no downloaded project is imported.
These checks do not establish any navigation benchmark performance.
"""
import json, math
from pathlib import Path

def released_batch_pattern(expert_ids, values):
    # Reconstruct only assignment and final call order seen in SkillNav.
    features=[]
    for expert_id,value in zip(expert_ids,values):
        selected_expert=expert_id
        features.append(value)
    return [x+10*selected_expert for x in features]

single=[released_batch_pattern([e],[x])[0] for e,x in zip([1,2],[0,0])]
mixed=released_batch_pattern([1,2],[0,0])
assert single==[10,20] and mixed==[20,20]
result={
 'test_kind':'Independent toy and algebra checks, not project execution or benchmark results',
 'skillnav_batch_assignment':{
  'setup':'Two distinct toy heads, same features; assignment order mirrors inspected lines647-664 then838.',
  'single_item_outputs':single,'mixed_batch_outputs':mixed,
  'finding':'The inspected call pattern can violate per-item expert routing; actual trained heads and paper runs were not evaluated.'},
 'mapnav_adjusted_metric':{
  'setup':{'terminal_distance_m':3.1,'optimal_length_m':5,'executed_length_m':20},
  'standard_3m_success':0,'standard_3m_spl':0,
  'relaxed_3_2m_spl_with_path_weight':5/20,
  'released_helper_adjusted_spl':1,
  'finding':'Even under a relaxed success radius, setting adjusted SPL to1 discards path efficiency.'},
 'susa_sigmoid_then_softmax':{
  'assumption':'Four branch weights are softmax(sigmoid(z)), finite real logits z.',
  'maximum_pairwise_ratio_supremum':math.e,
  'maximum_single_weight_supremum':math.e/(math.e+3),
  'minimum_single_weight_infimum':1/(1+3*math.e),
  'finding':'The gate cannot arbitrarily suppress a branch through these weights alone; branch-logit scale remains another degree of freedom.'},
 'polarmem_weighted_conflict_example':{
  'setup':'For a positive query, released mixed-modality score is HAS count minus2*NOT_HAS count.',
  'compatible_candidate':{'has':2,'not_has':0,'score':2},
  'conflicting_candidate':{'has':5,'not_has':1,'score':3},
  'finding':'Weighted scoring can rank a conflicting candidate above a compatible candidate; unlike lexicographic conflict-first ordering.'}
}
out=Path(__file__).with_name('isolated_logic_results.json')
out.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
