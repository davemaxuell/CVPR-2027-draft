"""CPU check of paired scene reporting, including unequal dataset/scene counts."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
from navigation_four.common import EVALUATION as RUN, now, sha, write
from tools.finalize_four_evaluation import scene_estimates


def main():
    rows = [{'scene':scene,'dataset':dataset,'success':v,'spl':v/2,'ndtw':v/4}
            for scene,dataset,values in [('a','r2r',[0.,1.]),('b','r2r',[1.]),('a','rxr',[1.]),('b','rxr',[0.,0.])]
            for v in values]
    weights=np.array([[1.,1.],[3.,0.],[0.,3.],[2.,1.]])
    expected=np.array([.5,.75,.5,(3/5+2/4)/2])
    result=scene_estimates(rows,weights,['a','b'])
    np.testing.assert_allclose(result,expected[:,None]*np.array([1.,.5,.25])[None],rtol=0,atol=1e-15)
    # Shared resamples make paired identical systems exactly zero regardless of order.
    for shift in range(7):
        shuffled=rows[shift:]+rows[:shift]
        assert np.array_equal(scene_estimates(shuffled,weights,['a','b'])-result,np.zeros((4,3)))
    write(RUN/'check_reporting.json',{'status':'passed','completed_utc':now(),'no_E_data_used':True,
        'unequal_scene_and_dataset_counts_macro_formula_verified':True,'seven_identical_system_pairings_zero':True,
        'source_sha256':sha(ROOT/'tools/finalize_four_evaluation.py'),'check_source_sha256':sha(__file__)})
    print('Four-model repeated-E reporting checks passed',flush=True)


if __name__=='__main__':main()
