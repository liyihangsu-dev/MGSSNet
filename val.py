"""
Validation script - evaluates trained models and generates result CSVs
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import nibabel as nib
import pandas as pd
from glob import glob

from model import build_cssr_model
from data import build_dataset
from utils import Metrics, set_logger, beijing_time
from options import args


class CHAOSDataset(Dataset):
    """CHAOS dataset for generalization testing"""
    def __init__(self, args, data_root_dir):
        self.data_dir = os.path.join(data_root_dir, 'CHAOS/chaos')
        self.image_size = args.image_size
        self.k = args.upscale
        self.samples = []

        t1_dir = os.path.join(self.data_dir, 't1')
        t2_dir = os.path.join(self.data_dir, 't2')

        for patient_id in os.listdir(t1_dir):
            t1_path = os.path.join(t1_dir, patient_id)
            t2_path = os.path.join(t2_dir, patient_id)
            if os.path.isdir(t1_path) and os.path.isdir(t2_path):
                t1_files = [f for f in os.listdir(t1_path) if f.endswith('.nii.gz')]
                t2_files = [f for f in os.listdir(t2_path) if f.endswith('.nii.gz')]
                for t1_file in t1_files:
                    prefix = t1_file.replace('.nii.gz', '')
                    t2_file = f"{prefix}.nii.gz"
                    if t2_file in t2_files:
                        self.samples.append({
                            't1_path': os.path.join(t1_path, t1_file),
                            't2_path': os.path.join(t2_path, t2_file)
                        })

        print(f'CHAOS dataset: {len(self.samples)} subjects')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        t1_vol = np.asanyarray(nib.load(sample['t1_path']).dataobj).astype(np.float32)
        t2_vol = np.asanyarray(nib.load(sample['t2_path']).dataobj).astype(np.float32)

        d = min(t1_vol.shape[2], t2_vol.shape[2])
        mid_idx = d // 2

        t1_slice = t1_vol[:, :, mid_idx]
        t2_slice = t2_vol[:, :, mid_idx]

        t1_slice = (t1_slice - t1_slice.mean()) / (t1_slice.std() + 1e-8)
        t2_slice = (t2_slice - t2_slice.mean()) / (t2_slice.std() + 1e-8)

        h, w = self.image_size, self.image_size

        t1_resized = cv2.resize(t1_slice, (w, h), interpolation=cv2.INTER_LINEAR)
        t2_gt = cv2.resize(t2_slice, (w, h), interpolation=cv2.INTER_LINEAR)
        lr_size = self.image_size // self.k
        t2_lr = cv2.resize(t2_gt, (lr_size, lr_size), interpolation=cv2.INTER_LINEAR)
        t2_lr = cv2.resize(t2_lr, (w, h), interpolation=cv2.INTER_NEAREST)

        return {
            'I_ref': torch.from_numpy(t1_resized).unsqueeze(0),
            'I_tar_lr': torch.from_numpy(t2_lr).unsqueeze(0),
            'I_tar_gt': torch.from_numpy(t2_gt).unsqueeze(0),
            'patient_id': sample['t1_path'].split('/')[-2]
        }


def find_checkpoint(log_dir, upscale):
    """Find the best checkpoint in log_dir"""
    checkpoints = glob(f"{log_dir}/*.pth")
    if not checkpoints:
        return None
    # Try to find best PSNR checkpoint
    best = None
    best_psnr = -1
    for ckpt in checkpoints:
        if 'PSNR' in ckpt:
            try:
                psnr = float(ckpt.split('PSNR_')[1].split('.')[0]) / 10000
                if psnr > best_psnr:
                    best_psnr = psnr
                    best = ckpt
            except:
                pass
    if best is None:
        best = checkpoints[0]
    return best


def evaluate_model(model, dataloader, device):
    """Evaluate model on dataset, returns metrics dict"""
    model.eval()
    psnr_vals = []
    ssim_vals = []
    lpips_vals = []

    with torch.no_grad():
        for data_item in dataloader:
            I_ref = data_item['I_ref'].to(device)
            I_tar_lr = data_item['I_tar_lr'].to(device)
            I_tar_gt = data_item['I_tar_gt'].to(device)

            I_hat = model(I_tar_lr, I_ref)

            psnr = Metrics.cal_psnr(I_hat, I_tar_gt).item()
            ssim = Metrics.cal_ssim(I_hat, I_tar_gt).item()
            lpips = Metrics.cal_lpips(I_hat, I_tar_gt).item()

            psnr_vals.append(psnr)
            ssim_vals.append(ssim)
            lpips_vals.append(lpips)

    return {
        'PSNR': np.mean(psnr_vals),
        'SSIM': np.mean(ssim_vals),
        'LPIPS': np.mean(lpips_vals)
    }


def generate_results_csv():
    """Generate result CSVs for all datasets and scales"""
    base_dir = './logs'

    # Model list
    models = ['Restormer', 'McASSR', 'A2CDic', 'CDDPE', 'DINet', 'MTrans', 'SANet']
    datasets = ['BraTSReg', 'IXI', 'CHAOS']
    scales = [2, 4, 8]

    for dataset in datasets:
        rows = []
        for model in models:
            row = {'Model': model}
            for scale in scales:
                log_dir = f'{base_dir}/{dataset}x{scale}/{model}x{scale}'
                checkpoint = find_checkpoint(log_dir, scale)

                if checkpoint is None:
                    row[f'x{scale}_PSNR'] = 'N/A'
                    row[f'x{scale}_SSIM'] = 'N/A'
                    row[f'x{scale}_LPIPS'] = 'N/A'
                    continue

                # Load model
                args.model = model
                args.upscale = scale
                args.dataset = dataset
                args.batch_size = 1
                args.image_size = 64

                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                m, _, _, _ = build_cssr_model(args)
                m.to(device)

                state = torch.load(checkpoint, map_location=device)
                m.load_state_dict(state['net'])
                m.eval()

                # Build dataset
                if dataset == 'CHAOS':
                    data_root = '/root/autodl-tmp/datasets'
                    chao_ds = CHAOSDataset(args, data_root)
                    dl = DataLoader(chao_ds, batch_size=1)
                else:
                    _, dl = build_dataset(args)

                # Evaluate
                metrics = evaluate_model(m, dl, device)
                row[f'x{scale}_PSNR'] = f"{metrics['PSNR']:.2f}"
                row[f'x{scale}_SSIM'] = f"{metrics['SSIM']:.4f}"
                row[f'x{scale}_LPIPS'] = f"{metrics['LPIPS']:.4f}"

                del m
                torch.cuda.empty_cache()

            rows.append(row)

        df = pd.DataFrame(rows)
        csv_path = f'mat/results/{dataset}_results.csv'
        df.to_csv(csv_path, index=False)
        print(f'Saved: {csv_path}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='BraTSReg')
    parser.add_argument('--upscale', type=int, default=4)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./val_results')
    args2 = parser.parse_args()

    # If specific model/dataset given, just evaluate that
    if args2.model:
        args.model = args2.model
        args.upscale = args2.upscale
        args.dataset = args2.dataset
        args.batch_size = 1
        args.image_size = 64

        device = torch.device('cuda:0')

        # Build model
        model, _, _, _ = build_cssr_model(args)
        model.to(device)

        # Find checkpoint
        if args2.log_dir:
            ckpt = find_checkpoint(args2.log_dir, args2.upscale)
        else:
            ckpt = find_checkpoint(f'./logs/{args.dataset}x{args.upscale}/{args.model}x{args.upscale}', args2.upscale)

        print(f'Loading checkpoint: {ckpt}')
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state['net'])
        model.eval()

        # Build dataloader
        if args2.dataset == 'CHAOS':
            data_root = '/root/autodl-tmp/datasets'
            ds = CHAOSDataset(args, data_root)
            dl = DataLoader(ds, batch_size=1)
        else:
            _, dl = build_dataset(args)

        # Evaluate
        metrics = evaluate_model(model, dl, device)
        print(f"\nResults for {args.model} on {args.dataset} x{args.upscale}:")
        print(f"  PSNR:  {metrics['PSNR']:.2f} dB")
        print(f"  SSIM:  {metrics['SSIM']:.4f}")
        print(f"  LPIPS: {metrics['LPIPS']:.4f}")
    else:
        # Generate all CSVs
        generate_results_csv()
