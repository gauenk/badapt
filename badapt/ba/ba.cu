
#include <torch/types.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define THREADS_PER_BLOCK 512






/*********************************************************************************

        
                  Get those boundaries


**********************************************************************************/



static __global__
void get_boundary_k(bool* bndy, const int* mask,
                      int H,int W, int K, bool no_corners){

    // -- get index --
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (H*W)) return;

    // -- get (height,width) index --
    int h = index / W;
    int w = index % W;

    // -- read map index --
    int label = mask[index];

    // -- read four edges --
    int alt_index;
    bool is_neq = false;
    bool any_neq = false;
    int count = 0;
    for(int kh = -K; kh <= K; kh++){
      for(int kw = -K; kw <= K; kw++){
        if (no_corners && abs(kh) == 1 && abs(kw) == 1){ continue; }

        // -- get neigh location --
        int ph = h + kh;
        int pw = w + kw;
        bool valid_h = (ph >= 0) and (ph <= (H-1));
        bool valid_w = (pw >= 0) and (pw <= (W-1));
        if (not(valid_h and valid_w)){ continue; }

        // -- check if label matches --
        alt_index = ph * W + pw;
        is_neq = mask[alt_index] != label;
        any_neq = any_neq || is_neq;

      }
    }

    bndy[index] = any_neq;
}


torch::Tensor get_boundary(torch::Tensor mask, int K, bool no_corners) {

    // -- check inputs --
    CHECK_INPUT(mask);
    auto H = mask.size(0); 
    auto W = mask.size(1);
    int NumPixels = H * W;

    // -- label boundaries --
    auto bndy = torch::zeros({H,W}, torch::dtype(torch::kBool).device(mask.device()));
    int nthreads = 512;
    int nblocks = (NumPixels - 1)/nthreads + 1;
    get_boundary_k<<<nblocks,nthreads>>>(bndy.data_ptr<bool>(),mask.data_ptr<int>(), H,W, K, no_corners);

    return bndy;
}








/*********************************************************************************


        Check if Changing a Query Point Changes the Number of Conn Comps


**********************************************************************************/


__device__ const bool simple_point_lut[256] = {
           0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0,
	       1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
	       0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,
	       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	       0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0,
	       0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0,
	       0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
	       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0,
	       0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0,
	       0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0};

__device__ inline int ischangeable_by_nbrs(bool* nbrs){
  int num = 0;
  int count_diff = 0;
#pragma unroll
   for (int i=7; i>=0; i--)
   {
      num <<= 1;
      if (nbrs[i]) num++;
      else count_diff++;
   }
   return num;//simple_point_lut[num];
}

__device__ inline
void set_mask(int NW,int N,int NE,int W,
              int E,int SW,int S,int SE,
              int label, bool oob, bool* nbrs){
  nbrs[0] = (label == NW) || ((NW==-1) and oob);
  nbrs[1] = (label == N)  || ((N ==-1) and oob);
  nbrs[2] = (label == NE) || ((NE==-1) and oob);
  nbrs[3] = (label == W)  || ((W ==-1) and oob);
  nbrs[4] = (label == E)  || ((E ==-1) and oob);
  nbrs[5] = (label == SW) || ((SW==-1) and oob);
  nbrs[6] = (label == S)  || ((S ==-1) and oob);
  nbrs[7] = (label == SE) || ((SE==-1) and oob);
  return;

}

static __global__
void check_is_changeable_k(bool* ischng, const int* mask, const int* query,
                           int Q, int H, int W, bool oob, bool ensure_bndy){

  // -- get index --
  int q_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (q_index >= Q) return;

  // -- unpack position --
  int h_i = query[2*q_index+0];
  int w_i = query[2*q_index+1];
  int mask_i = mask[h_i*W + w_i];
  int neighs[8] = {-1,-1,-1,-1,-1,-1,-1,-1};
  bool vec[8];

  // -- reading all neighboring mask values --
  int ix = 0;
  for(int dh=-1; dh<=1; dh++){
    for(int dw=-1; dw<=1; dw++){
      if ((dh==0)&&(dw==0)){ continue; }
      int h_prop = h_i + dh;
      int w_prop = w_i + dw;
      bool invalid_h = (h_prop < 0) or (h_prop>=H);
      bool invalid_w = (w_prop < 0) or (w_prop>=W);
      bool invalid = invalid_h or invalid_w;
      if (invalid){
        neighs[ix] = -1;
      }else{
        neighs[ix] = mask[h_prop*W + w_prop];
      }
      ix++;
    }
  }

  // -- check 0 and 1 --
  set_mask(neighs[0], neighs[1], neighs[2], neighs[3], neighs[4],
           neighs[5], neighs[6], neighs[7], 0, oob, vec);
  int check0 = ischangeable_by_nbrs(vec);
  set_mask(neighs[0], neighs[1], neighs[2], neighs[3], neighs[4],
           neighs[5], neighs[6], neighs[7], 1, oob, vec);
  int check1 = ischangeable_by_nbrs(vec);
  bool is_isolated = ((check0 == 255) and (mask_i == 1)) or ((check1 == 255) and (mask_i==0));

  // -- only if a face takes the value can we update to it --
  bool any_zero = (neighs[1]==0) or (neighs[3] == 0) or (neighs[4] == 0) or (neighs[6] == 0);
  bool any_one  = (neighs[1]==1) or (neighs[3] == 1) or (neighs[4] == 1) or (neighs[6] == 1);

  // -- assign --
  bool is_simple0;
  bool is_simple1 = simple_point_lut[check1];  // we don't care about preserving topology of bkgnd
  bool is_chng0;
  bool is_chng1;
  bool is_chng;
  if (ensure_bndy){
    // ? why not is_simple0?... since "1" is marked for removing, then we want to see
    is_chng0 = (any_zero and (mask_i==1)) ? is_simple1 : false;
    is_chng1 = (any_one  and (mask_i==0)) ? true : false; // as long as there is a connection
    is_chng = is_chng0 or is_chng1 or is_isolated;
  }else{
    is_simple0 = simple_point_lut[check0];
    is_chng0 = any_zero ? (is_simple0 or (check0==255)) : true;
    is_chng1 = any_one  ? (is_simple1 or (check1==255)) : true;
    is_chng  = (is_chng0 and is_chng1) or is_isolated;
  }

  ischng[q_index] = is_chng;

}

torch::Tensor check_is_changeable(torch::Tensor mask, torch::Tensor query,
                                  bool oob, bool ensure_bndy) {

    // -- check inputs --
    CHECK_INPUT(mask);
    CHECK_INPUT(query);
    int H = mask.size(0);
    int W = mask.size(1);
    int Q = query.size(0);
    int _two = query.size(1);
    assert(_two == 2);

    // -- check if updating each query point is valid --
    int nthreads = 512;
    int nblocks = (Q - 1)/nthreads + 1;
    auto opt = torch::dtype(torch::kBool).device(mask.device());
    auto ischng = torch::zeros({Q},opt);
    check_is_changeable_k<<<nblocks, nthreads>>>(ischng.data_ptr<bool>(),mask.data_ptr<int>(),
                                                 query.data_ptr<int>(), Q, H, W,
                                                 oob, ensure_bndy);
    return ischng;

}




/*********************************************************************************


        Compute the ratio of the log Potts Term


**********************************************************************************/


static __global__
void log_potts_ratio_k(int* lpotts, const int* mask, const int* query,
                       int Q, int H, int W){

  // -- get index --
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= Q) return;

  // -- unpack position --
  int h_i = query[2*index+0];
  int w_i = query[2*index+1];
  
  // -- reading all neighboring mask values --
  int summ = 0;
  for(int dh=-1; dh<=1; dh++){
    for(int dw=-1; dw<=1; dw++){

      if ((dh==0)&&(dw==0)){ continue; }
      int h_prop = h_i + dh;
      int w_prop = w_i + dw;

      bool invalid_h = (h_prop < 0) or (h_prop>=H);
      bool invalid_w = (w_prop < 0) or (w_prop>=W);
      bool invalid = invalid_h or invalid_w;

      if (not invalid){
        summ += 1 - 2*mask[h_prop*W + w_prop];
      }

    }
  }

  lpotts[index] = summ; // compute \sum_{j in 8-pt neighs of i} 1-2*s_j
}



torch::Tensor
log_potts_ratio(torch::Tensor mask, torch::Tensor query) {

    // -- check inputs --
    CHECK_INPUT(mask);
    CHECK_INPUT(query);
    int H = mask.size(0); 
    int W = mask.size(1);
    int Q = query.size(0);
    int _two = query.size(1);
    assert(_two == 2);

    // -- count edges per boundary point --
    int nthreads = 512;
    int nblocks = (Q - 1)/nthreads + 1;
    auto opt = torch::dtype(torch::kInt32).device(mask.device());
    auto lpotts = torch::zeros({Q,},opt);
    log_potts_ratio_k<<<nblocks, nthreads>>>(lpotts.data_ptr<int>(),
                                             mask.data_ptr<int>(),
                                             query.data_ptr<int>(),
                                             Q, H, W);
    return lpotts;
}




/*********************************************************************************


      Compute a bool map indicating where Strands are in the binary mask


**********************************************************************************/


__device__ inline bool is_strand(bool* nbrs) {
    // Opposite neighbor pairs (8-connected, center excluded):
    // Layout:  0 1 2
    //          3   4
    //          5 6 7

    // bool pair_diag_tl_br = nbrs[0] && nbrs[7]; // top-left  <-> bottom-right
    // bool pair_vert       = nbrs[1] && nbrs[6]; // top       <-> bottom
    // bool pair_diag_tr_bl = nbrs[2] && nbrs[5]; // top-right <-> bottom-left
    // bool pair_horiz      = nbrs[3] && nbrs[4]; // left      <-> right
    // bool pair_horiz_off0 = nbrs[3] && nbrs[2];
    // bool pair_horiz_off1 = nbrs[3] && nbrs[7];
    // bool pair_horiz_off2 = nbrs[4] && nbrs[0];
    // bool pair_horiz_off3 = nbrs[4] && nbrs[5];

    // "3" and any that are NOT {0,5}
    // ANY pair that are not adjacent or on the same axis...
    /********
             { 0: {1,2,3,5},
               1: {0,2},
               2: {0,1,4,7},
               3: {0,5},
               4: {2,7},
               5: {0,3,6,7}
               6: {5,7}
               7: {5,6,2,4},
             }
    ***************/

    bool is_strand_b = false;

    is_strand_b = is_strand_b or (nbrs[0] & nbrs[2]);
    is_strand_b = is_strand_b or (nbrs[0] & nbrs[4]);
    is_strand_b = is_strand_b or (nbrs[0] & nbrs[5]);
    is_strand_b = is_strand_b or (nbrs[0] & nbrs[6]);
    is_strand_b = is_strand_b or (nbrs[0] & nbrs[7]);

    is_strand_b = is_strand_b or (nbrs[1] & nbrs[3]);
    is_strand_b = is_strand_b or (nbrs[1] & nbrs[4]);
    is_strand_b = is_strand_b or (nbrs[1] & nbrs[5]);
    is_strand_b = is_strand_b or (nbrs[1] & nbrs[6]);
    is_strand_b = is_strand_b or (nbrs[1] & nbrs[7]);

    is_strand_b = is_strand_b or (nbrs[2] & nbrs[0]);
    is_strand_b = is_strand_b or (nbrs[2] & nbrs[3]);
    is_strand_b = is_strand_b or (nbrs[2] & nbrs[5]);
    is_strand_b = is_strand_b or (nbrs[2] & nbrs[6]);
    is_strand_b = is_strand_b or (nbrs[2] & nbrs[7]);

    // Layout:  0 1 2
    //          3   4
    //          5 6 7

    is_strand_b = is_strand_b or (nbrs[3] & nbrs[1]);
    is_strand_b = is_strand_b or (nbrs[3] & nbrs[2]);
    is_strand_b = is_strand_b or (nbrs[3] & nbrs[4]);
    is_strand_b = is_strand_b or (nbrs[3] & nbrs[6]);
    is_strand_b = is_strand_b or (nbrs[3] & nbrs[7]);

    is_strand_b = is_strand_b or (nbrs[4] & nbrs[0]);
    is_strand_b = is_strand_b or (nbrs[4] & nbrs[1]);
    is_strand_b = is_strand_b or (nbrs[4] & nbrs[3]);
    is_strand_b = is_strand_b or (nbrs[4] & nbrs[5]);
    is_strand_b = is_strand_b or (nbrs[4] & nbrs[6]);

    // Layout:  0 1 2
    //          3   4
    //          5 6 7

    is_strand_b = is_strand_b or (nbrs[5] & nbrs[0]);
    is_strand_b = is_strand_b or (nbrs[5] & nbrs[1]);
    is_strand_b = is_strand_b or (nbrs[5] & nbrs[2]);
    is_strand_b = is_strand_b or (nbrs[5] & nbrs[4]);
    is_strand_b = is_strand_b or (nbrs[5] & nbrs[7]);

    is_strand_b = is_strand_b or (nbrs[6] & nbrs[0]);
    is_strand_b = is_strand_b or (nbrs[6] & nbrs[1]);
    is_strand_b = is_strand_b or (nbrs[6] & nbrs[2]);
    is_strand_b = is_strand_b or (nbrs[6] & nbrs[3]);
    is_strand_b = is_strand_b or (nbrs[6] & nbrs[4]);

    is_strand_b = is_strand_b or (nbrs[7] & nbrs[0]);
    is_strand_b = is_strand_b or (nbrs[7] & nbrs[1]);
    is_strand_b = is_strand_b or (nbrs[7] & nbrs[2]);
    is_strand_b = is_strand_b or (nbrs[7] & nbrs[3]);
    is_strand_b = is_strand_b or (nbrs[7] & nbrs[5]);

    // // Count how many neighbors are active (to detect branching)
    int count = 0;
    for (int i = 0; i < 8; i++) count += (int)nbrs[i];

    // A strand: exactly one line passes through, no extra branches
    return (is_strand_b && (count <= 2));

}


static __global__
void find_strands_kernel(const int* bmask, bool* strands, int H, int W){

  // -- get index --
  int SIZE = H*W;
  int HW = H*W;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= SIZE) return;

  // -- unpack position --
  int h_i = index / W;
  int w_i = index % W;
  int neighs[8];
  bool vec[8];
  int curr_b = bmask[index];
  
  // -- reading all neighboring spix values --
  int K = 1;
  int curr_ix = 0;
  for(int dh=-K; dh<=K; dh++){
    for(int dw=-K; dw<=K; dw++){

      if ((dh==0)&&(dw==0)){ continue; }
      int h_prop = h_i + dh;
      int w_prop = w_i + dw;

      bool invalid_h = (h_prop < 0) or (h_prop>=H);
      bool invalid_w = (w_prop < 0) or (w_prop>=W);
      bool invalid = invalid_h or invalid_w;

      if (invalid){
        neighs[curr_ix] = -1;
      }else{
        neighs[curr_ix] = bmask[h_prop*W + w_prop];
      }
      curr_ix++;
    }
  }

  // -- mark true if its a member of a strand --
  set_mask(neighs[0], neighs[1], neighs[2], neighs[3],
           neighs[4], neighs[5], neighs[6], neighs[7], curr_b, true, vec);
  strands[index] = is_strand(vec);
}


torch::Tensor
find_strands(torch::Tensor bmask) {

    // -- check inputs --
    CHECK_INPUT(bmask);
    int H = bmask.size(0); 
    int W = bmask.size(1);
    int SIZE = H*W;

    // -- count edges per boundary point --
    int nthreads = 512;
    int nblocks = (SIZE - 1)/nthreads + 1;
    auto opt = torch::dtype(torch::kBool).device(bmask.device());
    auto strands = torch::zeros({H,W},opt);
    find_strands_kernel<<<nblocks, nthreads>>>(bmask.data_ptr<int>(),
                                               strands.data_ptr<bool>(), H, W);
    return strands;
}


/*********************************************************************************


      Compute a bool map indicating where Islands are in the binary mask


**********************************************************************************/

static __global__
void find_islands_kernel(const int* bmask, bool* islands, int H, int W){

  // -- get index --
  int SIZE = H*W;
  int HW = H*W;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= SIZE) return;

  // -- unpack position --
  int h_i = index / W;
  int w_i = index % W;
  int curr = bmask[index];
  
  // -- reading all neighboring spix values --
  int K = 5;
  bool is_island = false;
  int count = 0;

  // spiral out from center
  for(int level=1; level<=K; level++){

    count = 0;
    for(int dh=-level; dh<=level; dh++){
      for(int dw=-level; dw<=level; dw++){

        if ((abs(dh)!=level) and (abs(dw)!=level)){ continue; }

        int h_prop = h_i + dh;
        int w_prop = w_i + dw;

        bool invalid_h = (h_prop < 0) or (h_prop>=H);
        bool invalid_w = (w_prop < 0) or (w_prop>=W);
        bool invalid = invalid_h or invalid_w;
        if (invalid){ continue; }
        
        int neigh = bmask[h_prop*W + w_prop];
        count += (neigh == curr);
      }
    }

    is_island = is_island or ( (count <= 1) or ( (count <= 2) and (level>1) ) );
    
  }

  islands[index] = is_island;
}


torch::Tensor
find_islands(torch::Tensor bmask) {

    // -- check inputs --
    CHECK_INPUT(bmask);
    int H = bmask.size(0); 
    int W = bmask.size(1);
    int SIZE = H*W;

    // -- count edges per boundary point --
    int nthreads = 512;
    int nblocks = (SIZE - 1)/nthreads + 1;
    auto opt = torch::dtype(torch::kBool).device(bmask.device());
    auto islands = torch::zeros({H,W},opt);
    find_islands_kernel<<<nblocks, nthreads>>>(bmask.data_ptr<int>(),
                                               islands.data_ptr<bool>(),
                                               H, W);
    return islands;
}









/*********************************************************************************


                              PyBind11


**********************************************************************************/



namespace py = pybind11;
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_boundary",&get_boundary,"Bool map of the boundary pixels.",
        py::arg("mask"), py::arg("K")=1, py::arg("no_corners")=true);
  m.def("check_is_changeable",&check_is_changeable,"Check which query points can be updated.",
        py::arg("mask"),py::arg("query"),py::arg("oob")=true,py::arg("ensure_bndy")=false);
  m.def("log_potts_ratio",&log_potts_ratio,"Compute the ratio of a log Potts term");
  m.def("find_strands",&find_strands,"Find strands in a binary mask.");
  m.def("find_islands",&find_islands,"Find islands in a binary mask.");
}
