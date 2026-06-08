from ast import arg
from turtle import up
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torch.optim.lr_scheduler import StepLR

def build_cssr_model(args):
   model = None
   loss_func = None
   if args.model.startswith('Restormer'):
      from models import Restormer
      model = Restormer()

   elif args.model.startswith('DemoModel'):
      from models import DemoModel
      model = DemoModel()

   elif args.model.startswith('FuLoRA'):
      from models import FuLoRA
      model = FuLoRA()

   elif args.model.startswith('URestormer'):
      from models import URestormer
      model = URestormer()

   elif args.model.startswith('XiaomiMM'):
      from models.XiaomiMM.XiaomiMM import XiaomiMM
      model = XiaomiMM()

   elif args.model.startswith('McASSR'):
      from models.McASSR.Network import MC_ARSR
      model = MC_ARSR()

   elif args.model.startswith('A2CDic'):
      from models.A2CDic.A2CDic import A2CDic
      model = A2CDic()

   elif args.model.startswith('CDDPE'):
      from models.CDDPE.CDDPE import CDDPE
      model = CDDPE()

   elif args.model.startswith('DINet'):
      from models.DINet.DINet import DINet
      model = DINet()

   elif args.model.startswith('MTrans'):
      from models.MTrans.MTrans import MTrans
      model = MTrans()

   elif args.model.startswith('SANet'):
      from models.SANet.SANet import SANet
      model = SANet()

   # elif args.model.startswith('CMSF'):
   #    from models.CMSF.CMSF import CMSF
   #    model = CMSF()

   elif args.model.startswith('DyFusion'):
      from models.DyFusion.DyFusion import DyFusion
      model = DyFusion()

   elif args.model.startswith('PSSR'):
      from models.PSSR import PSSR, grs_total_loss
      model = PSSR(in_c=1, dim=64, scale=args.upscale, depth=8)
      # Use custom GRS loss instead of default L1Loss
      loss_func = 'grs_total_loss'

   # Default loss if not set
   if loss_func is None:
      loss_func = nn.L1Loss()
      args.loss_func = 'L1Loss'
   else:
      args.loss_func = 'GRS'

   if args.model.startswith('FuLoRA'):
       lora_params = [p for p in model.parameters() if p.requires_grad]
       optimizer = torch.optim.AdamW(lr=args.lr, params=lora_params, weight_decay=0.01)
       scheduler = None
   else:
       optimizer = torch.optim.AdamW(lr=args.lr, params=model.parameters(), weight_decay=1e-4)
       scheduler = None
   args.optimizer = 'AdamW'
   args.scheduler = 'Fixed LR (no scheduler)'

   return model,loss_func,optimizer,scheduler