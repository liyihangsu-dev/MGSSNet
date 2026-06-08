import os
import torch
import nibabel as nib
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import torchvision.utils as vutils
from dataclasses import dataclass
import math

# ========================== 兼容旧版 PyTorch 的高斯模糊 ==========================
def gaussian_blur(x, kernel_size=11, sigma=2.0):
    k = kernel_size
    x = x.unsqueeze(0)  # [B, C, H, W]
    coords = torch.linspace(-(k//2), k//2, k, device=x.device)
    xc, yc = torch.meshgrid(coords, coords, indexing='ij')
    gaussian = torch.exp(-(xc**2 + yc**2) / (2 * sigma**2))
    gaussian = gaussian / gaussian.sum()
    gaussian = gaussian.view(1,1,k,k).repeat(x.size(1),1,1,1)
    x = F.conv2d(x, gaussian, padding=k//2, groups=x.size(1))
    return x.squeeze(0)

class IXIDataset(Dataset):
    def __init__(self, args, data_root_dir, is_train=True, train_ratio=0.8):
        self.t1_dir = os.path.join(data_root_dir, "IXI-T1")
        self.t2_dir = os.path.join(data_root_dir, "IXI-T2")
        self.image_size = args.image_size
        self.k = args.upscale

        def get_patient_id(filename):
            return filename.rsplit('-', 1)[0]

        t1_files = {get_patient_id(f): f for f in os.listdir(self.t1_dir) if f.endswith('.nii.gz')}
        t2_files = {get_patient_id(f): f for f in os.listdir(self.t2_dir) if f.endswith('.nii.gz')}

        common_ids = sorted(set(t1_files.keys()) & set(t2_files.keys()))
        split_idx = int(len(common_ids) * train_ratio)

        if is_train:
            self.patient_ids = common_ids[:split_idx]
        else:
            self.patient_ids = common_ids[split_idx:]

        self.t1_files = t1_files
        self.t2_files = t2_files

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        t1_path = os.path.join(self.t1_dir, self.t1_files[patient_id])
        t2_path = os.path.join(self.t2_dir, self.t2_files[patient_id])

        t1_vol = np.asanyarray(nib.load(t1_path).dataobj).astype(np.float32)
        t2_vol = np.asanyarray(nib.load(t2_path).dataobj).astype(np.float32)

        d = min(t1_vol.shape[2], t2_vol.shape[2])
        mid_idx = d // 2
        t1_image = t1_vol[:, :, mid_idx]
        t2_image = t2_vol[:, :, mid_idx]

        t1_image = torch.from_numpy(t1_image).unsqueeze(0)
        t2_image = torch.from_numpy(t2_image).unsqueeze(0)

        t1_image = F.interpolate(t1_image.unsqueeze(0), size=(self.image_size,self.image_size), mode='bicubic', align_corners=False).squeeze(0)
        t2_image = F.interpolate(t2_image.unsqueeze(0), size=(self.image_size,self.image_size), mode='bicubic', align_corners=False).squeeze(0)

        t1_lr_image = gaussian_blur(t1_image, kernel_size=9, sigma=5.0)
        t1_lr_image = F.interpolate(t1_lr_image.unsqueeze(0), size=(self.image_size//self.k, self.image_size//self.k), mode='bicubic', align_corners=False)
        t1_lr_image = F.interpolate(t1_lr_image, size=(self.image_size,self.image_size), mode='bicubic', align_corners=False).squeeze(0)

        return {
            'I_ref': t2_image,
            'I_tar_lr': t1_lr_image,
            'I_tar_gt': t1_image
        }

class BraTSRegDataset(Dataset):
    def __init__(self, args, data_root_dir, is_train=False):
        self.args = args
        self.image_size = args.image_size
        self.upscale = args.upscale
        self.is_train = is_train

        if is_train:
            # self.data_dir = os.path.join(data_root_dir, "BraTSReg_Train")
            self.data_dir = os.path.join(data_root_dir, "BraTSReg_Val")
        else:
            self.data_dir = os.path.join(data_root_dir, "BraTSReg_Val")

        self.case_folders = sorted([
            f for f in os.listdir(self.data_dir)
            if f.startswith("BraTSReg_") and os.path.isdir(os.path.join(self.data_dir, f))
        ])

        self.samples = []
        for case_name in self.case_folders:
            num_index = case_name.split('_')[1]
            t2_file = f'BraTSReg_{num_index}_00_0000_t2.nii.gz'
            t1ce_file = f'BraTSReg_{num_index}_00_0000_t1ce.nii.gz'

            t2_path = os.path.join(self.data_dir, case_name, t2_file)
            t1ce_path = os.path.join(self.data_dir, case_name, t1ce_file)

            self.samples.append({
                "t1ce_path": t1ce_path,
                "t2_path": t2_path,
                "case": case_name
            })

        print(f"✅ 加载 {'训练集' if is_train else '验证集'}：{len(self.samples)} 个样本")

    def __len__(self):
        return len(self.samples)

    def normalize(self, img):
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return img

    def __getitem__(self, idx):
        sample = self.samples[idx]
        t1ce_path = sample["t1ce_path"]
        t2_path = sample["t2_path"]

        t1_vol = np.asarray(nib.load(t1ce_path).dataobj).astype(np.float32)
        t2_vol = np.asarray(nib.load(t2_path).dataobj).astype(np.float32)

        d = min(t1_vol.shape[2], t2_vol.shape[2])
        mid_idx = d // 2
        t1_img = t1_vol[:, :, mid_idx]
        t2_img = t2_vol[:, :, mid_idx]

        t1_img = torch.from_numpy(t1_img).unsqueeze(0)
        t2_img = torch.from_numpy(t2_img).unsqueeze(0)

        t1_img = F.interpolate(t1_img.unsqueeze(0), size=(self.image_size, self.image_size), mode='bicubic', align_corners=False).squeeze(0)
        t2_img = F.interpolate(t2_img.unsqueeze(0), size=(self.image_size, self.image_size), mode='bicubic', align_corners=False).squeeze(0)

        t1_img = self.normalize(t1_img)
        t2_img = self.normalize(t2_img)

        # ✅ 使用自定义高斯模糊
        t1_lr = gaussian_blur(t1_img, kernel_size=11, sigma=2.0)
        t1_lr = F.interpolate(t1_lr.unsqueeze(0), scale_factor=1/self.upscale, mode='bicubic', align_corners=False)
        t1_lr = F.interpolate(t1_lr, size=(self.image_size, self.image_size), mode='bicubic', align_corners=False).squeeze(0)

        return {
            "I_ref": t2_img,
            "I_tar_lr": t1_lr,
            "I_tar_gt": t1_img
        }

class CHAOSDataset(Dataset):
    def __init__(self, args, data_root_dir):
        self.data_dir = os.path.join(data_root_dir, "chaos")
        self.t1_dir = os.path.join(self.data_dir, "t1")
        self.t2_dir = os.path.join(self.data_dir, "t2")
        self.image_size = args.image_size
        self.k = args.upscale
        self.samples = []

        patient_ids = sorted(os.listdir(self.t1_dir))
        t1_files_by_patient = {p: sorted([f for f in os.listdir(os.path.join(self.t1_dir, p)) if f.endswith('.npz') and 't1_inphase' in f])
                               for p in patient_ids if os.path.isdir(os.path.join(self.t1_dir, p))}
        t2_files_by_patient = {p: set(os.listdir(os.path.join(self.t2_dir, p)))
                               for p in patient_ids if os.path.isdir(os.path.join(self.t2_dir, p))}

        for patient, t1_files in t1_files_by_patient.items():
            t2_files = t2_files_by_patient.get(patient, set())
            for t1_file in t1_files:
                idx = t1_file.rsplit('_', 1)[-1].replace('.npz', '')
                t2_file = f"patient{patient}_t2_{idx}.npz"
                if t2_file in t2_files:
                    self.samples.append({
                        't1_path': os.path.join(self.t1_dir, patient, t1_file),
                        't2_path': os.path.join(self.t2_dir, patient, t2_file)
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        t1_data = np.load(sample['t1_path'])
        t2_data = np.load(sample['t2_path'])
        t1_image = t1_data['img']
        t2_image = t2_data['img']
        t1_image = torch.from_numpy(t1_image).unsqueeze(0)
        t2_image = torch.from_numpy(t2_image).unsqueeze(0)

        t1_image = F.interpolate(t1_image.unsqueeze(0), size=(self.image_size,self.image_size), mode='bicubic', align_corners=False).squeeze(0)
        t2_image = F.interpolate(t2_image.unsqueeze(0), size=(self.image_size,self.image_size), mode='bicubic', align_corners=False).squeeze(0)

        t1_lr_image = gaussian_blur(t1_image, kernel_size=9, sigma=5.0)
        t1_lr_image = F.interpolate(t1_lr_image.unsqueeze(0), size=(self.image_size//self.k, self.image_size//self.k), mode='bicubic', align_corners=False)
        t1_lr_image = F.interpolate(t1_lr_image, size=(self.image_size,self.image_size), mode='bicubic', align_corners=False).squeeze(0)

        return {
            'I_ref': t2_image,
            'I_tar_lr': t1_lr_image,
            'I_tar_gt': t1_image
        }

def build_dataset(args):
    data_root_dir = None
    train_dataset = None
    test_dataset = None

    if args.dataset == 'BraTSReg':
        data_root_dir = '/root/autodl-tmp/datasets/BraTSReg/'
        test_dataset = BraTSRegDataset(args, data_root_dir=data_root_dir, is_train=False)
        train_dataset = BraTSRegDataset(args, data_root_dir=data_root_dir, is_train=True)
    elif args.dataset == 'IXI':
        data_root_dir = '/root/autodl-tmp/datasets/IXI'
        test_dataset = IXIDataset(args, data_root_dir=data_root_dir, is_train=False)
        train_dataset = IXIDataset(args, data_root_dir=data_root_dir, is_train=True)
    elif args.dataset == 'CHAOS':
        data_root_dir = '/root/autodl-tmp/datasets/CHAOS'
        test_dataset = CHAOSDataset(args, data_root_dir=data_root_dir)
        train_dataset = None
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    if train_dataset is not None:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size * 4, drop_last=False)
        return train_dataloader, test_dataloader
    else:
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size * 4, drop_last=False)
        return None, test_dataloader

@dataclass
class Args:
    image_size: int = 224
    upscale: int = 4
    dataset: str = 'BraTSReg'
    batch_size: int = 1

# if __name__ == '__main__':
#     args = Args()
#     dataset_ = BraTSRegDataset(args, data_root_dir='/root/autodl-tmp/datasets/BraTSReg/')
#     print(len(dataset_))

#     batch_data = dataset_[3]
#     I_t1 = batch_data['I_tar_gt']
#     I_t2 = batch_data['I_ref']
#     I_lr = batch_data['I_tar_lr']

#     vutils.save_image(I_t1, 'I_t1.png')
#     vutils.save_image(I_t2, 'I_t2.png')
#     vutils.save_image(I_lr, 'I_lr.png')
    