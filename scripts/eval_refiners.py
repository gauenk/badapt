"""

   Evaluate Refiners

"""

import os
import pandas as pd
from pathlib import Path
from badapt.eval import eval_refiner


def read_cache(root,rname,dname):
    fname = root / ("%s_%s.csv" % (rname,dname))
    if not fname.exists(): return None
    return pd.read_csv(fname)

def write_cache(res,root,rname,dname):
    fname = root / ("%s_%s.csv" % (rname,dname))
    res.to_csv(fname)


# def dev_code():

#     import torch as th

#     from badapt.deform import deform_mask
#     from badapt.data import get_eval_dataset
#     from badapt.ba_triton import find_islands,find_strands

#     import sclr
#     from sclr.isr_boundary import remove_strands


#     data = get_eval_dataset("thinobjects",'val')

#     sample = data[0]
#     image  = sample['image'].cuda(non_blocking=True)
#     gt = (sample['label'].round() > 128.0).int().cuda(non_blocking=True)
#     if gt.shape[0] == 0: return False
#     assert(gt.shape[0] == 1)
#     gt = gt[0]

#     coarse = deform_mask(gt,0.10,0.10,0.80)
#     refined = coarse

#     i0 = sclr.sclr_cuda.find_islands(refined[None,:])[0]
#     # i0 = sclr.sclr_cuda.find_strands(refined[None,:])[0]

#     i1 = find_islands(refined)
#     # i1 = find_strands(refined)
#     print("subset? ",th.all(i0[i1]))
#     print("subset? ",th.all(i1[i0]))

#     print(i0.shape,i1.shape)
#     print(type(i0),i0.dtype)
#     print(type(i1),i1.dtype)
#     print(th.sum(i0 != i1),i0.sum(),i1.sum())
#     exit()


def main():


    print("[PID: %s] Running Evaluation Script." % (str(os.getpid())))
    # dev_code()

    cache_root = Path("output/eval_cache/")
    # rnames = ['bsam','hqsam-refiner','sam-refiner']
    # dnames = ["thinobjects","big","dis5k","hrsod","uhrsd",]
    rnames = ['bsam',]
    dnames = ["thinobjects",]

    results = []
    for rname in rnames:
        for dname in dnames:

            res = read_cache(cache_root,rname,dname)
            if res is None:
                if rname == 'bsam':
                    res = eval_refiner(rname,dname,0.00,[12,9,3],[4,2,1])
                    # res = eval_refiner(rname,dname,0.00,[12+9+3,],[1,])
                    # res = eval_refiner(rname,dname,0.00,[10,],[1,])
                elif rname in ['hqsam-refiner','sam-refiner']:
                    res = eval_refiner(rname,dname)
                write_cache(res,cache_root,rname,dname)

            results.append(res)

    results = pd.concat(results)
    print(results)


if __name__ == "__main__":
    main()
