import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== DWT/IDWT ====================
class DWT(nn.Module):
    """Discrete Wavelet Transform using Haar wavelets"""
    def forward(self, x):
        B, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0, "Input must be divisible by 2"
        x = x.reshape(B, C, H // 2, 2, W // 2, 2)
        # LL, LH, HL, HH
        ll = x[:, :, :, 0, :, 0]
        lh = x[:, :, :, 0, :, 1]
        hl = x[:, :, :, 1, :, 0]
        hh = x[:, :, :, 1, :, 1]
        return ll, lh, hl, hh


class IDWT(nn.Module):
    """Inverse Discrete Wavelet Transform using Haar wavelets"""
    def forward(self, ll, lh, hl, hh):
        B, C, h, w = ll.shape
        x = torch.zeros(B, C, h * 2, w * 2, device=ll.device, dtype=ll.dtype)
        x[:, :, 0::2, 0::2] = ll
        x[:, :, 0::2, 1::2] = lh
        x[:, :, 1::2, 0::2] = hl
        x[:, :, 1::2, 1::2] = hh
        return x


# ==================== Attention Modules ====================
class SE(nn.Module):
    """Squeeze-and-Excitation block"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


# ==================== SemBlock ====================
class SemBlock(nn.Module):
    """
    Semantic Block with DWT-based frequency decoupling,
    Semantic Mixture for high-frequency features,
    and Semantic Alignment for low-frequency features.
    """
    def __init__(self, dim):
        super().__init__()
        self.dwt = DWT()
        self.idwt = IDWT()

        # Semantic Mixture: process high-frequency components
        self.high_conv = nn.Conv2d(dim * 3, dim * 3, 1, bias=False)
        self.se_high = SE(dim * 3)

        # Semantic Alignment: process low-frequency component
        self.low_conv = nn.Conv2d(dim, dim, 3, 1, 1, bias=False)
        self.pix_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),  # DWConv
            nn.PixelShuffle(1) if False else nn.Identity(),  # placeholder
        )

        # Actually use PixelShuffle for channel to spatial mapping
        self.pix_map = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.PixelShuffle(2),  # reduce channels by 4, increase spatial by 2
            nn.Conv2d(dim // 4, dim, 3, 1, 1, bias=False),
        )

        # Alignment module: concat and conv
        self.align_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
        )

        # Output projection
        self.out_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
        )

        # Downsample to match spatial dimensions
        self.down = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            nn.AvgPool2d(2, 2),
        )

    def forward(self, sem_feat, pix_feat):
        """
        Args:
            sem_feat: semantic feature from previous SemBlock or head, shape [B, C, H, W]
            pix_feat: pixel feature from DSPM, shape [B, C, H, W]
        Returns:
            next_sem_feat: refined semantic feature, shape [B, C, H/2, W/2]
        """
        # DWT decomposition
        ll, lh, hl, hh = self.dwt(sem_feat)
        # ll: [B, C, H/2, W/2], lh/hl/hh: [B, C, H/2, W/2]

        # --- Semantic Mixture (high-frequency) ---
        # Concatenate 3 high-freq components
        high_feat = torch.cat([lh, hl, hh], dim=1)  # [B, 3C, H/2, W/2]
        high_feat = self.high_conv(high_feat)
        high_feat = self.se_high(high_feat)

        # --- Semantic Alignment (low-frequency) ---
        # Upsample low-freq to match pix_feat spatial size, then process
        target_h, target_w = pix_feat.shape[2:]
        low_up = F.interpolate(ll, size=(target_h, target_w), mode='bilinear', align_corners=False)
        low_up = self.low_conv(low_up)

        # Process pix_feat with depthwise separable conv
        pix_dw = self.pix_map(pix_feat)  # [B, C, H, W] -> [B, C, H/2, W/2]

        # Align low-freq with pix_feat
        align_input = torch.cat([low_up, pix_dw], dim=1)
        ll_aligned = self.align_conv(align_input)

        # Upsample back to H/2 x W/2 for IDWT
        ll_aligned = F.interpolate(ll_aligned, size=(ll.shape[2], ll.shape[3]), mode='bilinear', align_corners=False)

        # --- Reconstruction via IDWT ---
        sem_rec = self.idwt(ll_aligned, lh, hl, hh)

        # Output with residual
        out = self.out_conv(sem_rec)

        # Downsample to H/2 x W/2 for next stage
        out = self.down(out)

        return out


# ==================== LoRA ====================
class LoRA(nn.Module):
    """
    Low-Rank Adaptation module.
    For a frozen weight W0, we add (alpha/r) * B * A where A: [r, C] and B: [C, r]
    """
    def __init__(self, dim, rank=4, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.rank = rank
        # A: down-project, B: up-project
        self.A = nn.Parameter(torch.randn(rank, dim), requires_grad=False)  # frozen init
        self.B = nn.Parameter(torch.zeros(dim, rank), requires_grad=True)  # trainable

    def forward(self, x):
        # W0 + (alpha/r) * B * A
        return F.conv2d(F.conv2d(x, self.A.unsqueeze(-1).unsqueeze(-1), groups=self.rank),
                        self.B.unsqueeze(-1).unsqueeze(-1)) * (self.alpha / self.rank)

    def reset_parameters(self):
        nn.init.normal_(self.A, std=1.0 / self.rank)
        nn.init.zeros_(self.B)


# ==================== DSPM ====================
class DSPM(nn.Module):
    """
    Dynamic Semantic-Pixel Context Mixing Module.
    Uses dual LoRA branches to model cross-modal correlations
    and generates adaptive convolution kernels for context aggregation.
    """
    def __init__(self, dim, groups=8, kernel_size=3, rank=4):
        super().__init__()
        self.dim = dim
        self.groups = groups
        self.k = kernel_size

        # Dual LoRA branches
        self.lora_s = LoRA(dim, rank=rank)
        self.lora_p = LoRA(dim, rank=rank)

        # Q, K projection layers (frozen backbone)
        self.Wq = nn.Conv2d(dim, dim, 1, bias=False)
        self.Wk = nn.Conv2d(dim, dim, 1, bias=False)

        # Dynamic kernel generation
        self.Wd = nn.Sequential(
            nn.Conv2d(groups, groups * (kernel_size ** 2), 1, bias=False),
        )

        # Output convolution
        self.out_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),  # depthwise
            nn.Conv2d(dim, dim, 1, bias=False),
        )

        # Disable gradients for Wq, Wk initially
        for p in self.Wq.parameters():
            p.requires_grad = False
        for p in self.Wk.parameters():
            p.requires_grad = False

    def forward(self, sem_feat, pix_feat):
        """
        Args:
            sem_feat: semantic feature, shape [B, C, H, W]
            pix_feat: pixel feature, shape [B, C, H, W]
        Returns:
            next_pix_feat: refined pixel feature, shape [B, C, H, W]
        """
        # LoRA adaptation
        fs = sem_feat + self.lora_s(sem_feat)
        fp = pix_feat + self.lora_p(pix_feat)

        # Q, K generation
        B, C, H, W = fs.shape
        g = self.groups
        c_per_g = C // g

        q = self.Wq(fs).reshape(B, g, c_per_g, H * W).permute(0, 1, 3, 2)  # [B, g, HW, c/g]
        k = self.Wk(fp).reshape(B, g, c_per_g, H * W).permute(0, 1, 3, 2)  # [B, g, HW, c/g]

        # Affinity matrix
        scale = (c_per_g) ** 0.5
        attn = F.softmax(torch.bmm(q, k.transpose(-2, -1)) / scale, dim=-1)  # [B, g, HW, HW]

        # Dynamic kernel: [B, g, HW, k*k] -> [B, g, k*k, HW]
        kernel = self.Wd(attn.permute(0, 1, 3, 2)).reshape(B, g, self.k * self.k, H, W)

        # Adaptive context aggregation via dynamic convolution
        pad_fp = F.pad(fp, [self.k // 2] * 4, mode='replicate')
        agg = torch.zeros_like(fp)

        for i in range(self.k):
            for j in range(self.k):
                agg = agg + kernel[:, :, i * self.k + j, :, :] * \
                      pad_fp[:, :, i:i + H, j:j + W]

        # Residual connection
        out = pix_feat + self.out_conv(agg)
        return out


# ==================== PixelCNN Head ====================
class PixelCNNHead(nn.Module):
    """Initial feature extraction with 7x7 conv"""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 7, 1, 3, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False),
        )

    def forward(self, x):
        return self.conv(x)


# ==================== PSSR Main Network ====================
class PSSR(nn.Module):
    """
    Pixel-level and Semantic-level Adaptive Super-Resolution Network.
    Parallel dual-branch architecture with Semantic Branch and Pixel Branch.
    """
    def __init__(self, in_c=1, dim=64, scale=4, depth=8):
        super().__init__()
        self.scale = scale
        self.depth = depth

        # Semantic branch: extract multi-scale semantic features
        self.sem_head = PixelCNNHead(in_c, dim)

        # Pixel branch: upsample and extract features
        self.pix_head = nn.Sequential(
            nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False),
            PixelCNNHead(in_c, dim)
        )

        # Stacked semantic and pixel blocks
        self.sem_blocks = nn.ModuleList([SemBlock(dim) for _ in range(depth)])
        self.dspm_blocks = nn.ModuleList([DSPM(dim) for _ in range(depth)])

        # Reconstruction head
        self.rec_head = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            nn.Conv2d(dim, in_c, 3, 1, 1, bias=False),
        )

    def forward(self, lr, ref):
        """
        Args:
            lr: low-resolution target image, [B, 1, H/s, W/s]
            ref: high-resolution reference image, [B, 1, H, W]
        Returns:
            sr: super-resolved image, [B, 1, H, W]
            sem_list: list of semantic features from each SemBlock
            pix_list: list of pixel features from each DSPM
        """
        # Initial feature extraction
        sem_feat = self.sem_head(ref)
        pix_feat = self.pix_head(lr)

        sem_list = []
        pix_list = []

        # Cross-stage feature interaction
        for i in range(self.depth):
            # Semantic branch generates guidance
            sem_feat = self.sem_blocks[i](sem_feat, pix_feat)

            # Pixel branch refines under semantic guidance
            pix_feat = self.dspm_blocks[i](sem_feat, pix_feat)

            sem_list.append(sem_feat)
            pix_list.append(pix_feat)

        # Reconstruction with global residual
        sr = self.rec_head(pix_feat)
        lr_up = F.interpolate(lr, scale_factor=self.scale, mode='bilinear', align_corners=False)
        sr = sr + lr_up

        return sr, sem_list, pix_list


# ==================== Gradient Reassignment Strategy Loss ====================
def grs_total_loss(pred, gt, sem_list, pix_list, lambda_k=1.0):
    """
    Gradient Reassignment Strategy loss.
    Main reconstruction loss + auxiliary semantic-pixel consistency loss.
    """
    # Primary reconstruction loss
    loss_rec = F.l1_loss(pred, gt)

    # Auxiliary loss: semantic-pixel consistency at each stage
    loss_aux = 0.0
    for s, p in zip(sem_list, pix_list):
        # s and p may have different spatial dims due to SemBlock downsampling
        # Resize to match
        if s.shape != p.shape:
            s = F.interpolate(s, size=p.shape[-2:], mode='bilinear', align_corners=False)
        loss_aux += F.l1_loss(s, p.detach())
    loss_aux = loss_aux / len(sem_list)

    # Total loss
    loss_total = loss_rec + lambda_k * loss_aux

    return loss_total, loss_rec, loss_aux


# ==================== Module Count Helper ====================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==================== Standalone Test ====================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PSSR(in_c=1, dim=64, scale=4, depth=8).to(device)

    print(f"Model params: {count_parameters(model):,}")

    # Test forward
    B, C, H, W = 2, 1, 256, 256
    lr = torch.randn(B, C, H // 4, W // 4, device=device)
    ref = torch.randn(B, C, H, W, device=device)
    gt = torch.randn(B, C, H, W, device=device)

    sr, sem_list, pix_list = model(lr, ref)
    print(f"Output shape: {sr.shape}, Expected: [{B}, {C}, {H}, {W}]")
    assert sr.shape == gt.shape, f"Shape mismatch: {sr.shape} vs {gt.shape}"

    # Test loss
    loss, loss_rec, loss_aux = grs_total_loss(sr, gt, sem_list, pix_list)
    print(f"Total loss: {loss.item():.4f} | Rec: {loss_rec.item():.4f} | Aux: {loss_aux.item():.4f}")

    # Backward pass
    loss.backward()
    print("Backward pass successful!")

    print("\n✅ PSSR model test passed!")