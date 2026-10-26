"""


    Triton is much slower the compiled CUDA kernels


"""

# -- cuda imports --
import badapt._ba_cuda as ba_cuda
get_boundary_cu = ba_cuda.get_boundary
check_is_changeable_cu = ba_cuda.check_is_changeable
log_potts_ratio_cu = ba_cuda.log_potts_ratio
find_islands_cu = ba_cuda.find_islands
find_strands_cu = ba_cuda.find_strands

# -- triton imports --
from badapt.ba import ba_triton
get_boundary_tr = ba_triton.get_boundary
check_is_changeable_tr = ba_triton.check_is_changeable
log_potts_ratio_tr = ba_triton.log_potts_ratio
find_islands_tr = ba_triton.find_islands
find_strands_tr = ba_triton.find_strands


def eval_get_boundary_kernel():
    pass

def eval_check_is_changeable():
    pass

def eval_log_potts_ratio():
    pass

def eval_find_islands():
    pass

def eval_find_strands():
    pass

if __name__ == "__main__":
    eval_get_boundary_kernel()
    eval_check_is_changeable()
    eval_log_potts_ratio()
    eval_find_islands()
    eval_find_strands()
