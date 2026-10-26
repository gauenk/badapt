"""

  The B-SAM Architecture

  logit_i := log[p(z_i=1|...)/p(z_i=0|...)] \approx BSAM(...) + log-Potts-ratio(...)

"""

import torch as th
from torch import nn
from einops import rearrange
import torch.nn.functional as F

from segment_anything import sam_model_registry
from segment_anything.modeling.common import LayerNorm2d
from badapt.sam.image_encoder import init_image_encoder
from badapt.sam.mask_decoder_hq import init_mask_decoder
from badapt.ba import boundary_adaptation

class BSAM(nn.Module):

    def __init__(self,checkpoint_fname,samhq=True):
        super().__init__()

        # -- sam base --
        ckpt_fname = "checkpoints/sam_vit_h_4b8939.pth"
        self.sam = {"sam": sam_model_registry["vit_h"](checkpoint=ckpt_fname)}
        self.sam['sam'].image_encoder = init_image_encoder("checkpoints/sam_hq_vit_h.pth")
        self.sam['sam'].mask_decoder = init_mask_decoder("checkpoints/sam_hq_vit_h.pth")

        # -- low-level feature extraction --
        self.stride0 = 8
        s0 = self.stride0
        Fll = 512
        self.iftr_net = nn.Sequential(
            nn.ConvTranspose2d(1280, Fll, kernel_size=2, stride=2),
            LayerNorm2d(Fll),
            nn.GELU(),
            nn.ConvTranspose2d(Fll, Fll, kernel_size=2, stride=2),
            LayerNorm2d(Fll),
            nn.GELU(),
        )
        self.proj_mask_cur  = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(16, 16), stride=(s0, s0), padding=0),
            LayerNorm2d(32), nn.GELU(),
        )
        self.proj_mask_prev = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(16, 16), stride=(s0, s0), padding=0),
            LayerNorm2d(32), nn.GELU(),
        )
        self.mask_mix = nn.Sequential(
            nn.Conv2d(32 + 32, 64, kernel_size=1, padding=0),
            LayerNorm2d(64), nn.GELU(),
        )

        # -- FilM Parameters --
        self.scales   = [4, 2, 1]
        self.ll_scale_gamma = nn.ParameterList([nn.Parameter(th.zeros(Fll)) for _ in self.scales])
        self.ll_scale_beta  = nn.ParameterList([nn.Parameter(th.zeros(Fll)) for _ in self.scales])
        self.mask_scale_gamma = nn.ParameterList([nn.Parameter(th.zeros(64)) for _ in self.scales])
        self.mask_scale_beta  = nn.ParameterList([nn.Parameter(th.zeros(64)) for _ in self.scales])

        # -- coordinate encoder --
        self.pos_encoder = FourierCoordEncoder(num_freqs=8, max_freq_log2=8)
        pos_dim = self.pos_encoder.out_dim

        # -- imnet --
        self.imnet = PointwiseNet(64 + pos_dim,Fll + pos_dim,32  + pos_dim,256,hyper_dim=32)
        self.reset()

        # -- load bsam checkpoint --
        state = th.load(checkpoint_fname)['model_state']
        translate = {"NarrativeImNet":"PointwiseNet","coord_encoder":"pos_encoder",
                     "ftrs0_proj":"iftr_net","FourierCoordEncoder_v2":"FourierCoordEncoder"}
        renamed_state = {}
        for old_k,v in state.items():
            new_k = old_k
            for old_str,new_str in translate.items():
                if old_str in old_k:
                    new_k = old_k.replace(old_str,new_str)
                    break
            renamed_state[new_k] = v
        self.load_state_dict(renamed_state,strict=True)
        
    def reset(self):
        self.gridI    = None
        self.gridS    = None
        self.gridM    = None

    @th.no_grad()
    def get_sam_features(self, image, mask):
        with th.no_grad():
            iftrs_raw,sftrs,context = run_sam(self.sam['sam'], image, mask)
        return iftrs_raw,sftrs,context

    def iftr_net_ms(self, iftrs, scale):
        """
        Mix cached native-res ftrs0_base with mask_cur and curr_mask,
        then apply scale FiLM so imnet knows the current multiscale context.

        ftrs0_base : (1, Fll, fH0, fW0)  — native-res, cached
        obj_masks  : (1, H_s, W_s)       — mask at scaled resolution
        curr_mask  : (1, H_s, W_s)       — previous step mask (same scale)
        ms_scale   : int                  — e.g. 16, 8, 4, 2, 1
        """

        # -- scale FiLM --
        scale_idx = self.scales.index(scale)
        gamma = self.ll_scale_gamma[scale_idx]
        beta  = self.ll_scale_beta[scale_idx]
        iftrs = (1 + gamma)[None, :, None, None] * iftrs \
            +      beta  [None, :, None, None]
        return iftrs  # (1, Fll, fH0, fW0)

    def mftr_net(self, coarse, refined, scale, HW):

        def _get_patchify_padding(self, imgs):
            stride = self.stride0
            H, W = imgs.shape[-2:]
            pad_h = (stride - H % stride) % stride
            pad_w = (stride - W % stride) % stride
            if pad_h > 0 or pad_w > 0:
                imgs = F.pad(imgs, (0, pad_w, 0, pad_h))
            return imgs, (pad_h, pad_w)

        def encode_mask(mask_s, proj):

            # nearest upsample to native res — block size = scale pixels
            mask_native = F.interpolate(
                mask_s[None, None].float(), size=HW, mode='nearest'
            ).squeeze(1)  # (1, H, W)
            mask_pad, _ = self._get_patchify_padding(mask_native[:, None].float())
            return proj(mask_pad)  # (1, 32, fH0, fW0)
    
        # -- encode and mix --
        mask_cur_ftrs  = encode_mask(coarse,  self.proj_mask_cur)
        refined_ftrs = encode_mask(refined,  self.proj_mask_prev)
        mftrs = self.mask_mix(
            th.cat([mask_cur_ftrs, refined_ftrs], dim=1)
        )

        # -- scale FiLM --
        scale_idx = self.scales.index(scale)
        gamma = self.mask_scale_gamma[scale_idx]
        beta  = self.mask_scale_beta[scale_idx]
        mftrs = (1 + gamma)[None, :, None, None] * mftrs \
            +   beta  [None, :, None, None]
        return mftrs  # (1, Fll, fH0, fW0)

    def _get_grids(self, HW, iHW, sHW, mHW, sI, sS, sM, scale, device):

        def make_grid(shape, scale, s, device=None):
            fH, fW = shape
            row_seq = (th.arange(fH, device=device).float() * s + (s - 1) / 2.0) / scale
            col_seq = (th.arange(fW, device=device).float() * s + (s - 1) / 2.0) / scale
            ret = th.stack(th.meshgrid(row_seq, col_seq, indexing='ij'), dim=-1)
            return ret  # [fH, fW, 2], values in native pixel space
        
        if self.gridI is None: # cached across BA steps
            self.gridI = make_grid(iHW, scale, sI, device=device)
            self.gridS = make_grid(sHW, scale, sS, device=device)
            self.gridM = make_grid(mHW,     1, sM, device=device)
        return self.gridI, self.gridS, self.gridM

    def forward(self, query, fI, fS, fM, context, HW):

        def read_points(ftrs,fgrids,scale,stride):
            fH, fW = ftrs.shape[-2:]
            row = ((query[:, 0]*scale) / stride).long().clamp(0, fH - 1) # native-res to ftr-res index
            col = ((query[:, 1]*scale) / stride).long().clamp(0, fW - 1)
            pos     = fgrids[row, col] # full grid @ ftr-res to (N,2) with values @ nativ-res
            rel_pos = query - pos # offset in native-res
            pos_enc = self.pos_encoder(rel_pos)   # (N, grid_ndim)
            pos_feat  = ftrs[0, :, row, col].T  # (N, Fll)
            inp = th.cat([pos_feat, pos_enc], dim=-1)   # (N, Fll+grid_ndim)
            return inp

        # -- prepare reading --
        device = fI.device
        scale = 1024 / max(HW)
        iHW,sHW,mHW = fI.shape[-2:],fS.shape[-2:],fM.shape[-2:]
        sI,sS,sM = 1024 / iHW[0], 1024 / sHW[0], self.stride0

        # -- get grids ( grid @ ftrs.shape but values @ native-image.shape ) --
        gI, gS, gM = self._get_grids(HW, iHW, sHW, mHW, sI, sS, sM, scale, device)
        
        # -- read pointwise info --
        qM = read_points(fM,gM,1,sM) # mask
        qI = read_points(fI,gI,scale,sI) # image
        qS = read_points(fS,gS,scale,sS) # semantic

        # -- decode --
        logits = self.imnet(qI, qS, qM, context.view(1, -1))  # (N,)
        return logits

class PointwiseNet(nn.Module):

    def __init__(self, m_dim, in0_dim, in1_dim, hidden, hyper_dim):
        super().__init__()

        # --- per-scale encoders (static weights) ---
        self.encM = nn.Sequential(nn.Linear(m_dim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU())
        self.enc0 = nn.Sequential(nn.Linear(in0_dim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU())
        self.enc1 = nn.Sequential(nn.Linear(in1_dim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU())
        self.hyper_proj = nn.Sequential(nn.Linear(hyper_dim, hidden), nn.ReLU())
        self.dec_proj = nn.Sequential(nn.Linear(hyper_dim, hidden), nn.ReLU())
        hyper_dim = hidden

        # --- per-scale FiLM conditioning ---
        self.film0 = FiLMLayer(hidden, hyper_dim)
        self.film1 = FiLMLayer(hidden, hyper_dim)

        # --- hyper-gated scale fusion ---
        self.scale_gate = nn.Sequential(
            nn.Linear(hyper_dim, hidden),
            nn.Sigmoid()
        )

        # --- post-fusion refinement ---
        self.fuse = FusionBlock(hidden)
        self.fused_for_dec = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())
        self.dec_film = FiLMLayer(hyper_dim, hidden)
        self.fused_for_hyper = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())
        self.hyper_film = FiLMLayer(hyper_dim, hidden)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, inp0, inp1, inpM, context):

        # lift features to latent space
        q = inp0.shape[0]
        context = self.hyper_proj(context)
        f0 = self.enc0(inp0) + self.encM(inpM)
        f0 = self.film0(f0, context)   # [bs, q, hidden]
        f1 = self.film1(self.enc1(inp1), context)   # [bs, q, hidden]

        # fusion of semantic and low-level features
        g = self.scale_gate(context).unsqueeze(1)   # [bs, 1, hidden]
        fused = g * f0 + (1 - g) * f1            # [bs, q, hidden]
        fused = fused.view(q,-1)
        fused = self.fuse(fused)

        # modulate sam-based pair with fused features
        dec = self.dec_proj(inp1[:,:32])
        dec_mod = self.dec_film(dec,self.fused_for_dec(fused))
        context_mod = self.hyper_film(context,self.fused_for_hyper(fused))
        logits = th.sum(dec_mod * context_mod,dim=-1) / (context_mod.shape[-1] ** 0.5)

        return logits


class FusionBlock(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.act = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.act(x + self.net(x))  # residual: preserve gated fusion signal


class FiLMLayer(nn.Module):
    def __init__(self, hidden_dim, hyper_dim):
        super().__init__()
        self.to_gamma_beta = nn.Linear(hyper_dim, hidden_dim * 2)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)

    def forward(self, x, hyper):
        # hyper: [bs, hyper_dim] → unsqueeze to broadcast over q
        gb = self.to_gamma_beta(hyper)       # [bs, 1, hidden*2]
        gamma, beta = gb.chunk(2, dim=-1)                 # [bs, 1, hidden] each
        return (1 + gamma) * x + beta                     # broadcasts over q

class FourierCoordEncoder(nn.Module):
    def __init__(self, num_freqs=8, max_freq_log2=3, include_input=True):
        super().__init__()
        self.include_input = include_input
        self.num_freqs = num_freqs
        freqs = 2.0 ** th.linspace(0, max_freq_log2, num_freqs)
        self.register_buffer('freqs', freqs)

    @property
    def out_dim(self):
        base = 2 * self.num_freqs * 2
        if self.include_input:
            base += 2
        return base

    def forward(self, rel_pos):
        """
        rel_pos: [..., 2]  normalized offset in [-1, 1] within the feature cell
        returns: [..., out_dim]
        """
        x_freq = rel_pos.unsqueeze(-1) * self.freqs   # [..., 2, num_freqs]
        parts = [x_freq.sin().flatten(-2),
                 x_freq.cos().flatten(-2)]
        if self.include_input:
            parts = [rel_pos] + parts
        return th.cat(parts, dim=-1)


@th.no_grad()
def run_sam(sam,image,mask):

    from badapt.sam.sam_refiner import sam_input_prepare
    from segment_anything.utils.transforms import ResizeLongestSide

    # -- setup image --
    device = mask.device
    H,W,C = image.shape
    img_size = sam.image_encoder.img_size

    # -- prepare image --
    img_size = sam.image_encoder.img_size
    resize = ResizeLongestSide(img_size)
    image = resize.apply_image(image.cpu().numpy())
    image = th.as_tensor(image, device=device)
    image = image.permute(2, 0, 1).contiguous()

    # -- run image encoder [slloow] --
    input_images = sam.preprocess(image)[None,:]
    enc,iftrs_raw = sam.image_encoder(input_images,return_early_ftrs=True)
    enc = enc.detach()
    iftrs_raw = iftrs_raw.detach()
    iftrs_raw = rearrange(iftrs_raw,'1 h w c -> 1 c h w')

    # -- get prompts --
    image_record = sam_input_prepare(image, mask[None,:], enc[0], resize, gamma=4.0, strength=30)[0]

    # -- run mask decoder --
    points = None
    if "point_coords" in image_record:
        points = (image_record["point_coords"], image_record["point_labels"])
    sparse_embeddings, dense_embeddings = sam.prompt_encoder(
        points=points,boxes=image_record.get("boxes", None),
        masks=image_record.get("mask_inputs", None),
    )

    # -- run mask decoder --
    image_pe = sam.prompt_encoder.get_dense_pe()
    fwd = sam.mask_decoder.forward_alt
    sftrs,context = fwd(enc,image_pe,sparse_embeddings,dense_embeddings,
                        multimask_output=False,hq_token_only=True,
                        interm_embeddings=iftrs_raw,return_upscaled_hq=True)

    return iftrs_raw,sftrs,context


def vizualize_grid_index():

    import numpy as np
    import matplotlib.pyplot as plt
    import scipy

    H = 20
    W = 20

    scale = 4 
    bndy = th.tensor([(0,0),(1,1),(3,4),]).numpy()
    query = scale*bndy+(scale-1)/2.0

    res = 8
    nH = res*H
    nW = res*W
    grid = th.stack(th.meshgrid(th.arange(nH),th.arange(nW),indexing='ij'),dim=-1).view(-1,2)
    grid = grid.numpy()
    img = np.zeros((nH,nW))
    plt.imshow(img,'viridis')


    # -- major points --
    og_bool = np.logical_and(grid[:,0]%res == 0,grid[:,1]%res == 0)
    major_grid = grid[og_bool]
    print(major_grid[:3]/res)
    _major_grid = grid[og_bool] + res//2
    plt.scatter(_major_grid[:,0],major_grid[:,1],s=20)

    # -- highlight nearest grid-point to query --
    dist = scipy.spatial.distance.cdist(query,grid/(1.0*res))
    inds = np.argmin(dist,axis=1)
    iq = grid[inds]+res//2
    plt.scatter(iq[:,0],iq[:,1],s=10)
    plt.savefig("vizualize_grid_index.png")



if __name__ == "__main__":
    vizualize_grid_index()
