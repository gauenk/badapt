"""

   Evaluate Refiners

"""

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

def main():

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
                elif rname in ['hqsam-refiner','sam-refiner']:
                    res = eval_refiner(rname,dname)
                write_cache(res,cache_root,rname,dname)

            results.append(res)

    results = pd.concat(results)
    print(results)


if __name__ == "__main__":
    main()
