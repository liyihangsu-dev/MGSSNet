import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse

class RepConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RepConv, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        return self.conv3x3(x) + self.conv1x1(x)

class RestormerBlock(nn.Module):
    def __init__(self, channels):
        super(RestormerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.qkv_dw = nn.Conv2d(channels * 3, channels * 3, 3, 1, 1, groups=channels * 3)
        self.project_out = nn.Conv2d(channels, channels, 1)
        
        self.ffn_in = nn.Conv2d(channels, channels * 2, 1)
        self.ffn_dw = nn.Conv2d(channels * 2, channels * 2, 3, 1, 1, groups=channels * 2)
        self.ffn_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        res = x
        
        x_norm = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        x_norm = self.norm1(x_norm).reshape(b, h, w, c).permute(0, 3, 1, 2)
        
        qkv = self.ffn_dw(self.qkv(x_norm))
        q, k, v = qkv.chunk(3, dim=1)
        
        q = q.reshape(b, c, h * w)
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w)
        
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        
        attn = torch.bmm(q, k.transpose(-1, -2))
        attn = F.softmax(attn, dim=-1)
        
        out = torch.bmm(attn, v).reshape(b, c, h, w)
        out = self.project_out(out) + res
        
        res_ffn = out
        out_norm = out.permute(0, 2, 3, 1).reshape(b, h * w, c)
        out_norm = self.norm2(out_norm).reshape(b, h, w, c).permute(0, 3, 1, 2)
        
        ffn = self.ffn_dw(self.ffn_in(out_norm))
        g1, g2 = ffn.chunk(2, dim=1)
        out = g1 * F.gelu(g2)
        out = self.ffn_out(out) + res_ffn
        return out

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))

class SCGAM(nn.Module):
    def __init__(self, channels):
        super(SCGAM, self).__init__()
        self.norm = nn.GroupNorm(4, channels)
        
        self.dwconv7 = nn.Sequential(nn.Conv2d(channels, channels, 7, 1, 3, groups=channels), nn.ReLU(inplace=True))
        self.dwconv5 = nn.Sequential(nn.Conv2d(channels, channels, 5, 1, 2, groups=channels), nn.ReLU(inplace=True))
        self.dwconv3 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, groups=channels), nn.ReLU(inplace=True))
        
        self.dwt = DWTForward(J=1, wave='haar', mode='zero')
        self.idwt = DWTInverse(wave='haar', mode='zero')
        
        self.g_conv = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1), nn.ReLU(inplace=True))
        
        self.post_dw7 = nn.Sequential(nn.Conv2d(channels, channels, 7, 1, 3, groups=channels), nn.ReLU(inplace=True))
        self.post_dw5 = nn.Sequential(nn.Conv2d(channels, channels, 5, 1, 2, groups=channels), nn.ReLU(inplace=True))
        self.post_dw3 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, groups=channels), nn.ReLU(inplace=True))
        
        self.concat_conv = nn.Conv2d(channels * 3, channels, 3, 1, 1)

    def forward(self, F_sem_i, F_g_next=None):
        res = F_sem_i
        F_sem_i = self.norm(F_sem_i)
        
        x7 = self.dwconv7(F_sem_i)
        x5 = self.dwconv5(F_sem_i)
        x3 = self.dwconv3(F_sem_i)
        
        yl7, yh7 = self.dwt(x7)
        yl5, yh5 = self.dwt(x5)
        yl3, yh3 = self.dwt(x3)
        
        if F_g_next is not None:
            g_feat = self.g_conv(F_g_next)
            g_yl, g_yh = self.dwt(g_feat)
            
            yl7 = yl7 + g_yl
            yl5 = yl5 + g_yl
            yl3 = yl3 + g_yl
            
        out7 = self.post_dw7(self.idwt((yl7, yh7)))
        out5 = self.post_dw5(self.idwt((yl5, yh5)))
        out3 = self.post_dw3(self.idwt((yl3, yh3)))
        
        fused = torch.cat([out7, out5, out3], dim=1)
        fused = self.concat_conv(fused)
        return fused + res

class SS2D(nn.Module):
    def __init__(self, channels):
        super(SS2D, self).__init__()
        self.proj = nn.Conv2d(channels, channels, 1)
        self.A_log = nn.Parameter(torch.log(torch.ones(channels, 1)))
        self.D = nn.Parameter(torch.ones(channels))

    def forward(self, x):
        b, c, h, w = x.shape
        x_proj = self.proj(x).view(b, c, h * w)
        delta = F.softmax(x_proj, dim=-1)
        A = -torch.exp(self.A_log)
        dA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(1))
        y = x_proj * dA.mean(dim=-1) + x_proj * self.D.unsqueeze(0).unsqueeze(-1)
        return y.view(b, c, h, w)

class SSMM(nn.Module):
    def __init__(self, channels):
        super(SSMM, self).__init__()
        self.norm = nn.GroupNorm(4, channels)
        self.dwconv5 = nn.Sequential(nn.Conv2d(channels, channels, 5, 1, 2, groups=channels), nn.ReLU(inplace=True))
        self.dwconv3 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, groups=channels), nn.ReLU(inplace=True))
        
        self.ss2d_shared = SS2D(channels)
        
        self.pconv = nn.Conv2d(channels, channels, 1)
        self.pnorm = nn.GroupNorm(4, channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x
        x = self.norm(x)
        
        branch5 = self.dwconv5(x)
        branch3 = self.dwconv3(x)
        
        feat_ss2d_up = self.ss2d_shared(branch5)
        feat_ss2d_down = self.ss2d_shared(branch3 + branch5)
        
        merged = feat_ss2d_up * feat_ss2d_down
        out = self.relu(self.pnorm(self.pconv(merged)))
        return out + res

class ISSAM(nn.Module):
    def __init__(self, channels):
        super(ISSAM, self).__init__()
        self.m1 = SSMM(channels)
        self.m2 = SSMM(channels)
        self.m3 = SSMM(channels)
        self.res_block = ResidualBlock(channels * 2)
        self.out_conv = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, F_sem_i, F_g_i):
        m1_out = self.m1(F_sem_i)
        f_g_mid = F_g_i + m1_out
        
        m2_out = self.m2(f_g_mid)
        f_sem_mid = F_sem_i * m2_out
        
        m3_out = self.m3(f_g_mid)
        f_sem_out = f_sem_mid + m3_out
        
        combined = torch.cat([f_sem_out, f_g_mid], dim=1)
        out = self.out_conv(self.res_block(combined))
        return out

class AdaptiveMultiGranularityFusion(nn.Module):
    def __init__(self, channels):
        super(AdaptiveMultiGranularityFusion, self).__init__()
        self.res1 = ResidualBlock(channels)
        self.res2 = ResidualBlock(channels)
        self.res3 = ResidualBlock(channels)
        
        self.dw1 = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels)
        self.dw2 = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels)
        self.dw3 = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels)
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.router = nn.Linear(channels * 3, 3)

    def forward(self, f1, f2, f3):
        b, c, h, w = f1.shape
        
        f1_res = self.res1(f1)
        f2_res = self.res2(f2)
        f3_res = self.res3(f3)
        
        f2_down = F.interpolate(f2_res, size=(h, w), mode='bilinear', align_corners=False)
        f3_down = F.interpolate(f3_res, size=(h, w), mode='bilinear', align_corners=False)
        
        f1_mid = f1_res + self.dw1(f2_down)
        f2_mid = f2_res + self.dw2(f1_res) + self.dw2(f3_res)
        f3_mid = f3_res + self.dw3(f2_res)
        
        f2_mid_down = F.interpolate(f2_mid, size=(h, w), mode='bilinear', align_corners=False)
        f3_mid_down = F.interpolate(f3_mid, size=(h, w), mode='bilinear', align_corners=False)
        
        f1_out = self.res1(f1_mid)
        f2_out = self.res2(f2_mid_down)
        f3_out = self.res3(f3_mid_down)
        
        gap1 = self.gap(f1_out).view(b, c)
        gap2 = self.gap(f2_out).view(b, c)
        gap3 = self.gap(f3_out).view(b, c)
        
        gate_feats = torch.cat([gap1, gap2, gap3], dim=1)
        weights = F.softmax(self.router(gate_feats), dim=-1)
        
        w1 = weights[:, 0].view(b, 1, 1, 1)
        w2 = weights[:, 1].view(b, 1, 1, 1)
        w3 = weights[:, 2].view(b, 1, 1, 1)
        
        fused = w1 * f1_out + w2 * f2_out + w3 * f3_out
        return fused

class MGSSNet(nn.Module):
    def __init__(self, in_channels=1, ref_channels=1, base_channels=64):
        super(MGSSNet, self).__init__()
        self.conv_ref = nn.Conv2d(ref_channels, base_channels, 3, 1, 1)
        self.res_ref1 = ResidualBlock(base_channels)
        self.res_ref2 = ResidualBlock(base_channels)
        self.res_ref3 = ResidualBlock(base_channels)
        
        self.conv_lr = nn.Conv2d(in_channels, base_channels, 3, 1, 1)
        self.rep_conv = RepConv(base_channels, base_channels)
        
        self.restormer1 = RestormerBlock(base_channels)
        self.restormer2 = RestormerBlock(base_channels)
        self.restormer3 = RestormerBlock(base_channels)
        
        self.scgam3 = SCGAM(base_channels)
        self.scgam2 = SCGAM(base_channels)
        self.scgam1 = SCGAM(base_channels)
        
        self.issam1 = ISSAM(base_channels)
        self.issam2 = ISSAM(base_channels)
        self.issam3 = ISSAM(base_channels)
        
        self.amf = AdaptiveMultiGranularityFusion(base_channels)
        self.recon = nn.Conv2d(base_channels, in_channels, 3, 1, 1)

    def forward(self, I_lr, I_ref):
        F_spa = self.conv_ref(I_ref)
        F_sem_ref1 = self.res_ref1(F_spa)
        F_sem_ref2 = self.res_ref2(F_sem_ref1)
        F_sem_ref3 = self.res_ref3(F_sem_ref2)
        
        F_sem_lr0 = self.conv_lr(I_lr)
        
        F_restormer1 = self.restormer1(F_sem_lr0)
        F_sem_lr1 = F.interpolate(F_restormer1, scale_factor=0.5, mode='bilinear', align_corners=False)
        
        F_restormer2 = self.restormer2(F_sem_lr1)
        F_sem_lr2 = F.interpolate(F_restormer2, scale_factor=0.5, mode='bilinear', align_corners=False)
        
        F_restormer3 = self.restormer3(F_sem_lr2)
        F_sem_lr3 = F.interpolate(F_restormer3, scale_factor=0.5, mode='bilinear', align_corners=False)
        
        F_g3 = self.scgam3(F_sem_lr3, None)
        
        F_g3_up = F.interpolate(F_g3, scale_factor=2.0, mode='bilinear', align_corners=False)
        F_g2 = self.scgam2(F_sem_lr2, F_g3_up)
        
        F_g2_up = F.interpolate(F_g2, scale_factor=2.0, mode='bilinear', align_corners=False)
        F_g1 = self.scgam1(F_sem_lr1, F_g2_up)
        
        F_g1_up2 = F.interpolate(F_g1, scale_factor=2.0, mode='bilinear', align_corners=False)
        F_g2_up4 = F.interpolate(F_g2, scale_factor=4.0, mode='bilinear', align_corners=False)
        F_g3_up8 = F.interpolate(F_g3, scale_factor=8.0, mode='bilinear', align_corners=False)
        
        F_ssa1 = self.issam1(F_sem_ref1, F_g1_up2)
        F_ssa2 = self.issam2(F_sem_ref2, F_g2_up4)
        F_ssa3 = self.issam3(F_sem_ref3, F_g3_up8)
        
        F_fused = self.amf(F_ssa1, F_ssa2, F_ssa3)
        
        F_rep = self.rep_conv(F_sem_lr0)
        out_feat = F_fused + F_rep
        
        I_hat = self.recon(out_feat)
        return I_hat