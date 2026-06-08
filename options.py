import argparse
import torch

def get_args():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="low-level Training Configuration")
    
    # --- Basic Configuration ---
    # Use 'choices' to restrict input and prevent typos
    parser.add_argument('--task', type=str, default="csmr", 
                        choices=["deblurring", "fusion", "denoising"], 
                        help='Select task type (default: deblurring)')
    parser.add_argument('--model', type=str, default='DemoModel', help='Model name')
    # --- Data Configuration ---
    parser.add_argument('--dataset', type=str, default='BraTSReg', help='Dataset name')
    parser.add_argument('--batch_size', type=int, default=4, help='Input batch size for training')
    parser.add_argument('--patch_size', type=int, default=64, help='Size of training patches')
    parser.add_argument('--image_size', type=int, default=224, help='Input image resolution')
    parser.add_argument('--upscale', type=int, default=4, help='Upscaling factor')
    parser.add_argument('--inp_channels', type=int, default=3, help='number of inpute image channels')
    # --- Training Configuration ---
    parser.add_argument('--epochs', type=int, default=50, help='Number of total epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--seed', type=int, default=3407, help='Random seed for reproducibility')
    parser.add_argument('--check_point', type=str, default=None, help='checkpoint file to resume training')
    # --- Advanced Usage: Boolean Flags ---
    # Uses 'store_true', so if the flag is present, the value is True
    parser.add_argument('--log', type=int, default=0, help='Enable logging (1) or disable (0)')
    # Define a '--test' flag to easily switch to evaluation mode
    parser.add_argument('--test', action='store_true', help='Run in test mode only (no training)')
    
    # --- Device Selection ---
    # Automatically selects 'cuda' if available, otherwise falls back to 'cpu'
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', 
                        help='Device to run the model on (cuda/cpu)')
    
    # --- Model Hyperparameters ---
    parser.add_argument('--dim', type=int, default=64, help='Model dimension')
    parser.add_argument('--depth', type=int, default=4, help='Network depth')
    
    # --- Saving Strategy ---
    parser.add_argument('--val_step', type=int, default=1, help='Validation frequency (in epochs)')
    parser.add_argument('--save_step', type=int, default=1, help='Model checkpoint saving frequency')

    # Parse arguments
    args = parser.parse_args()
    
    # --- Post-processing Logic ---
    # If '--test' flag is provided, override the training flag
    if args.test:
        args.is_train = 0
    else:
        args.is_train = 1
        
    return args

args = get_args()