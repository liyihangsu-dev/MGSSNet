import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.nn.init as init

import numpy as np
import logging
from datetime import datetime
from datetime import timedelta
from datetime import timezone
import sys
import os
import shutil
import time
import shutil
import copy
import cv2

from PIL import ImageDraw
from thop import profile, clever_format
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from PIL import Image
sys.path.append('../datasets')


from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.regression import MeanSquaredError
import lpips



def init_weights(m: nn.Module) -> None:
    """
    神经网络权重初始化函数
    
    根据不同层类型采用相应的初始化策略，使用 isinstance 进行精确类型判断
    
    参数:
        m: 待初始化的 PyTorch 模块
    """
    # 线性层（Linear/Bilinear）使用 Kaiming 均匀初始化
    if isinstance(m, (nn.Linear, nn.Bilinear)):
        nn.init.kaiming_uniform_(m.weight, a=2, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    # 卷积层使用 Kaiming 均匀初始化
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, a=2, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    # 归一化层使用均匀初始化
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
        nn.init.uniform_(m.weight, a=0, b=1)
        nn.init.zeros_(m.bias)

    # RNN Cell 层使用 Xavier 均匀初始化
    elif isinstance(m, (nn.RNNCell, nn.LSTMCell, nn.GRUCell)):
        nn.init.xavier_uniform_(m.weight_hh, gain=1)
        nn.init.xavier_uniform_(m.weight_ih, gain=1)
        nn.init.ones_(m.bias_hh)
        nn.init.ones_(m.bias_ih)

    # RNN 层使用 Xavier 均匀初始化
    elif isinstance(m, (nn.RNN, nn.LSTM, nn.GRU)):
        for w in m.all_weights:
            nn.init.xavier_uniform_(w[2].data, gain=1)
            nn.init.xavier_uniform_(w[3].data, gain=1)
            nn.init.ones_(w[0].data)
            nn.init.ones_(w[1].data)

    # Embedding 层使用 Kaiming 均匀初始化
    elif isinstance(m, nn.Embedding):
        nn.init.kaiming_uniform_(m.weight, a=2, mode='fan_in', nonlinearity='leaky_relu')



class Metrics:
    """量化指标计算对象
    """
    def __init__(self, logger):
        pass
    
    @staticmethod
    @torch.no_grad()
    def cal_psnr(pre_hsi, gt):
        if len(pre_hsi.shape) == 3:
            pre_hsi = pre_hsi.unsqueeze(0)
            gt = gt.unsqueeze(0)
        mse = torch.mean((pre_hsi - gt) ** 2, dim=[1, 2, 3])
        # 使用动态范围 (max - min) 作为 data_range，更适合医学图像
        data_range = gt.max() - gt.min()
        if data_range.item() == 0:
            return torch.tensor(float('inf'), device=pre_hsi.device)
        # 避免除零
        mse = torch.clamp(mse, min=1e-10)
        psnr = 10 * torch.log10(data_range ** 2 / mse)
        return psnr.mean()
            
    @staticmethod
    @torch.no_grad()
    def cal_ssim(pre_hsi, gt):
        if len(pre_hsi.shape) == 3:
            pre_hsi = pre_hsi.unsqueeze(0)
            gt = gt.unsqueeze(0)
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(gt.device)
        return ssim(pre_hsi, gt)
    
    @staticmethod
    @torch.no_grad()
    def cal_sam(pre_hsi, gt):
        assert gt.shape == pre_hsi.shape
        if len(pre_hsi.shape) == 3:
            pre_hsi = pre_hsi.unsqueeze(0)
            gt = gt.unsqueeze(0)
        gt = gt.permute(0, 2, 3, 1)
        pre_hsi = pre_hsi.permute(0, 2, 3, 1)
        dot_product = torch.sum(gt * pre_hsi, dim=-1)
        norm_reference = torch.norm(gt, dim=-1)
        norm_target = torch.norm(pre_hsi, dim=-1)
        cos_theta = dot_product / (norm_reference * norm_target + 1e-10)
        sam_map = torch.acos(cos_theta)
        sam = torch.mean(sam_map)*180/torch.pi
        return sam


    @staticmethod
    @torch.no_grad()
    def cal_ergas(pre_hsi, gt, up_scale=4):
        assert gt.shape == pre_hsi.shape
        if len(pre_hsi.shape) == 3:
            pre_hsi = pre_hsi.unsqueeze(0)
            gt = gt.unsqueeze(0)
        rmse_bands = torch.sqrt(torch.mean((gt - pre_hsi) ** 2, dim=(2, 3)))  # (B, C)
        mean_bands = torch.mean(gt, dim=(2, 3))  # (B, C)
        ergas = 100 / up_scale * torch.sqrt(torch.mean((rmse_bands / mean_bands) ** 2, dim=1))  # (B,)
        return ergas.mean().item()
    
    @staticmethod
    @torch.no_grad()
    def cal_mse(pre_hsi, gt):
        if len(pre_hsi.shape) == 3:
            pre_hsi = pre_hsi.unsqueeze(0)
            gt = gt.unsqueeze(0)
        mse = MeanSquaredError().to(gt.device)
        return mse(pre_hsi, gt)

    @staticmethod
    @torch.no_grad()
    def cal_lpips(pre_hsi, gt, net='alex'):
        """Calculate Learned Perceptual Image Patch Similarity (LPIPS)
        Uses lpips library. Lower is better (closer to 0 = identical).
        """
        if len(pre_hsi.shape) == 3:
            pre_hsi = pre_hsi.unsqueeze(0)
            gt = gt.unsqueeze(0)
        # lpips expects RGB [0,1] or [-1,1], normalize to [0,1] range
        pre_hsi_norm = (pre_hsi - pre_hsi.min()) / (pre_hsi.max() - pre_hsi.min() + 1e-8)
        gt_norm = (gt - gt.min()) / (gt.max() - gt.min() + 1e-8)
        # Convert gray to RGB by repeating
        if pre_hsi_norm.shape[1] == 1:
            pre_hsi_norm = pre_hsi_norm.repeat(1, 3, 1, 1)
            gt_norm = gt_norm.repeat(1, 3, 1, 1)
        lpips_model = lpips.LPIPS(net=net).to(gt.device)
        return lpips_model(pre_hsi_norm, gt_norm).mean()
    
def beijing_time():
    """获取北京时间

    Returns:
        str: 北京时间字符串
    """
    utc_now = datetime.utcnow().replace(tzinfo=timezone.utc)
    SHA_TZ = timezone(
        timedelta(hours=8),
        name='Asia/Shanghai',
    )
    beijing_now = utc_now.astimezone(SHA_TZ)
    fmt = '%Y-%m-%d-%H-%M-%S'
    now_fmt=beijing_now.strftime(fmt)
    return  now_fmt

def set_seed(seed=9999):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_logger(args):
    """获取日志的打印对象

    Args:
        model_name (str): 模型名字
        logger_dir (str): 模型打印的路径
        log_out (bool): 是否输出打印

    Returns:
        _type_: 日志打印对象
    """
    logger_dir = args.log_dir
    model_name = args.model.split('_')[0]
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')

    if args.log == 1:
        log_file = f"{logger_dir}/out.log"
        if not os.path.exists(log_file):
             with open(log_file, 'w') as F:
                 pass
        fileHandler = logging.FileHandler(log_file)
        fileHandler.setLevel(logging.INFO)
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)
        source_file = f'./models/{model_name}'
        target_file = f"{logger_dir}/{model_name}"
        shutil.copytree(source_file, target_file)
        
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.INFO) 
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)
    return logger

@torch.no_grad()
def test_speed(args, model):
    """Test model inference speed, FLOPs, parameter count, and generate parameter table.

    Robust implementation using fvcore.nn (primary) with fallback to thop.
    Falls back to manual counting if both libraries fail.

    Args:
        args: Global args object with device, image_size attributes
        model: Model to test

    Returns:
        inference_time (float): seconds per inference
        flops (float): FLOPs in billions (G)
        params (float): parameter count in millions (M)
        param_table (str): formatted parameter table
    """
    device = torch.device(args.device if hasattr(args, 'device') else 'cuda:0')

    # Deep copy and prepare model
    test_model = copy.deepcopy(model)
    test_model.eval()
    test_model.to(device)

    # Create dummy inputs matching expected signature
    dummy_input = torch.rand(1, 1, args.image_size, args.image_size).to(device)
    dummy_ref = torch.rand(1, 1, args.image_size, args.image_size).to(device)

    # ---- FLOPs and params using fvcore ----
    flops = 0.0
    params = 0.0
    param_table = "N/A"

    try:
        from fvcore.nn import FlopCountAnalysis, parameter_count_table
        analysis = FlopCountAnalysis(test_model, (dummy_input, dummy_ref))
        flops = analysis.total() / 1e9  # Convert to G
        params = sum(p.numel() for p in test_model.parameters()) / 1e6  # M params
        param_table = str(parameter_count_table(test_model))
    except Exception as fv_err:
        # ---- Fallback: try thop ----
        try:
            from thop import profile, clever_format
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                flops_dict, params_dict = profile(
                    test_model, inputs=(dummy_input, dummy_ref), verbose=False
                )
                if isinstance(flops_dict, tuple):
                    flops = flops_dict[0] / 1e9
                elif flops_dict is not None:
                    flops = float(flops_dict) / 1e9
                else:
                    flops = 0.0
                if isinstance(params_dict, tuple):
                    params = params_dict[0] / 1e6
                elif params_dict is not None:
                    params = float(params_dict) / 1e6
                else:
                    params = sum(p.numel() for p in test_model.parameters()) / 1e6
            param_table = f"Parameters: {params:.2f}M (thop)"
        except Exception as thop_err:
            # ---- Final fallback: manual counting ----
            params = sum(p.numel() for p in test_model.parameters()) / 1e6
            flops = 0.0
            param_table = f"Parameters: {params:.2f}M (manual)"

    # ---- Warm-up run ----
    with torch.no_grad():
        for _ in range(3):
            _ = test_model(dummy_input, dummy_ref)

    # ---- Measure inference time (average of 10 runs) ----
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()

    with torch.no_grad():
        for _ in range(10):
            _ = test_model(dummy_input, dummy_ref)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    inference_time = (time.time() - start_time) / 10.0

    # ---- Clean up ----
    del test_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return inference_time, flops, params, param_table



def get_cave_hsi_image(image_path):
    hsi = []
    for dir in sorted(os.listdir(image_path)):
        if dir.endswith('.png'):
            image_dir = os.path.join(image_path,dir)
            image_data = Image.open(image_dir)
            image_data = np.array(image_data)
            hsi.append(image_data)
    hsi = np.stack(hsi,axis=0) / 65535
    return hsi

def plot_tensor_image(image_data):
    """将torch图像转换为numpy图像
        (C, H, W) -> (H, W, C)
    Args:
        image_data (torch.tensor): 输入的图像数据 (C, H, W)

    Returns:
        np.array: 输出的numpy图像 (H, W, C)
    """
    if len(image_data.shape)==2:
        image_data = image_data.unsqueeze(0)
    return image_data.permute(1,2,0).detach().cpu().numpy()


def down_sample(hsi, down_scale=4):
    """对图像进行高斯模糊 + 下采样

    Args:
        hsi (_type_): 输入的高光谱图像 (C, H, W)
        down_scale (int, optional): _description_. Defaults to 4.

    Returns:
        _type_: 下采样后的高光谱图像 (C, H//down_scale, W//down_scale)
    """
    hsi = hsi.unsqueeze(0)
    down_hsi = torchvision.transforms.GaussianBlur(kernel_size=3, sigma=0.5)(hsi)
    down_hsi = F.interpolate(down_hsi,scale_factor=1/down_scale)
    down_hsi = down_hsi.squeeze(0)
    return down_hsi

def sam_errormap(reference_image, reconstructed_image):
    """
    绘制预测的图像和参考图像之间的SAM误差图

    Args:
        reference_image (np.ndarray): 用于计算SAM的参考图像 (H, W, C)
        reconstructed_image (np.ndarray): 预测的图像 (H, W, C)

    Returns:
        np.ndarray: (H, W)
    """
    height, width, num_bands = reference_image.shape
    errormap = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            # Extract pixel spectral vectors
            ref_spectrum = reference_image[i, j, :]  # Reference image pixel spectrum
            rec_spectrum = reconstructed_image[i, j, :]  # Reconstructed image pixel spectrum
            
            # Calculate spectral angle SAM
            numerator = np.dot(ref_spectrum, rec_spectrum)  # Dot product
            denominator = np.linalg.norm(ref_spectrum) * np.linalg.norm(rec_spectrum)  # Product of norms
            if denominator == 0:
                errormap[i, j] = 0  # Set to 0 if the norm is zero
            else:
                cos_theta = numerator / denominator
                cos_theta = np.clip(cos_theta, -1, 1)  # Clip cos_theta to the range [-1, 1]
                errormap[i, j] = np.arccos(cos_theta)  # Calculate angle in radians
    errormap = errormap * (180 / np.pi)  # Convert radians to degrees
    return errormap

def img_resize(img, size):
    """cv2.resize的封装，输入图像为(h,w,c)

    Args:
        img (np.array): 输入图像 (h,w,c)
        size (int): 输出图像大小

    Returns:
        np.array: 输出图像 (size,size,c)
    """
    img = img.transpose(1, 2, 0)
    # img = mmcv.imresize(img, size)
    img = cv2.resize(img,size)
    img = img.transpose(2, 0, 1)
    return img

def crop_patchs(image,strid,patch_size):
    """切割图像为patchs

    Args:
        image (np.array): 输入图像 (h,w,c)
        strid (int): 步长
        patch_size (int): patch大小

    Returns:
        list[np.array]: patchs (nums,patch_size,patch_size,c)
    """
    patchs = []
    c,h,w = image.shape
    for i in range(0,h-patch_size+1,strid):
        for j in range(0,w-patch_size+1,strid):
            patch = image[:,i:i+patch_size,j:j+patch_size]
            patchs.append(patch)
    return patchs


def draw_rec(image,x,y,h,w,scale=2,loc='left',line_width=4):
    """图像细节图 （画框）

    Args:
        image (np.array): 输入图像 (h,w,c)
        x (int): x坐标 (|)
        y (int): y坐标 (-)
        h (int): 高度
        w (int): 宽度
        scale (int): _description_. Defaults to 2.
        loc (str): _description_. Defaults to 'left'.
        line_width (int, optional): _description_. Defaults to 4.

    Returns:
        _type_: _description_
    """
    image = torch.from_numpy(image).permute(2,0,1).float()
    tar_region = image[:,x:x+h,y:y+w]
    _,H,W = image.shape
    scale_tar_region = F.interpolate(tar_region.unsqueeze(0),scale_factor=scale).squeeze(0)
    c,scale_h,scale_w = scale_tar_region.shape
    if loc == 'left':
        image[:,-scale_h:,:scale_w] = scale_tar_region
    elif loc == 'right':
        image[:,-scale_h:,-scale_w:] = scale_tar_region
    image = torchvision.transforms.ToPILImage()(image)
    draw = ImageDraw.Draw(image)
    draw.rectangle([y, x, y+w, x+h],
               outline=(255, 0, 0),
               width=line_width)
    if loc == 'left':
        draw.rectangle([0, H-scale_h-1, scale_w, H-1],
               outline=(255, 0, 0),
               width=line_width)
    elif loc == 'right':
        draw.rectangle([W-scale_w-1, H-scale_h-1, W-1, H-1],
               outline=(255, 0, 0),
               width=line_width)
    image = torchvision.transforms.ToTensor()(image)
    image = image.permute(1,2,0).numpy()
    return image