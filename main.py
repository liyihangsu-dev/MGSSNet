import torch
import logging
from utils import *
from model import *
from data import *
from train import *
import json
from options import args    


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed) 
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    train_dataloader,test_dataloader = build_dataset(args)
    model,loss_func,optimizer,scheduler  = build_cssr_model(args)

    model.apply(init_weights) # init model 
    model.to(device)

    train_func = train_cssr
    print(f">>> Using train_cssr for {args.model}")

    inference_time,flops,params,params_table = test_speed(args, model)
    if args.check_point is None:
        log_dir = f'./logs/{args.dataset}x{args.upscale}/{args.model}x{args.upscale},{beijing_time()},{os.getpid()}'
        
        if args.log == 1 and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        args.log_dir = log_dir
        args.start_epoch = 0

    logger = set_logger(args)
    logger.info(params_table)
    args.model_size = f"{params}M"
    args.inference_time = f"{(inference_time * 1000):.6f}ms"
    args.FLOPs = f"{flops}G"
    args.pid = os_id = os.getpid()
    logger.info(json.dumps(vars(args), indent=4))
    train_func(args, model, device, train_dataloader, test_dataloader,optimizer, loss_func, logger)