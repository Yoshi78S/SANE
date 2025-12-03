import torch
import numpy as np
import random
import argparse
import os
os.environ["OMP_NUM_THREADS"] = "20"


def parse_args():
    parser = argparse.ArgumentParser(description="Graphormer")
    parser.add_argument('--data_dir', type=str, default="data/")
    parser.add_argument('--data', type=str, default="epinions")
    parser.add_argument('--offset', type=float, default=4.0)
    parser.add_argument('--topks', type=str, default='[5,10,15,20]')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--lambda_reg', type=float, default=1e-4)
    parser.add_argument('--test_batch_size', type=int, default=1024)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--valid_interval', type=int, default=20)
    parser.add_argument('--stopping_step', type=int, default=10)
    parser.add_argument('--show_loss_interval', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--model', type=str, default="eig+path")
    parser.add_argument('--alpha', type=float, default=0.)
    parser.add_argument('--beta', type=float, default=1.)
    parser.add_argument('--eigs_dim', type=int, default=64)
    parser.add_argument('--sample_hop', type=int, default=4)
    parser.add_argument('--item_pair_tau', type=float, default=0.05)
    parser.add_argument('--item_pair_weight', type=float, default=0.005)
    parser.add_argument('--item_pair_sample_size', type=int, default=100000)
    
    # NiDen parameters (percentile-based thresholds)
    parser.add_argument('--niden_start_epoch', type=int, default=1500,
                        help='Epoch to start NiDen denoising (recommend >= 50 for warm-up)')
    parser.add_argument('--niden_rate_max', type=float, default=0.01,
                        help='Maximum removal rate (e.g., 0.05 = top 5%% edges removed)')
    parser.add_argument('--niden_rate_min', type=float, default=0.005,
                        help='Minimum removal rate (e.g., 0.01 = top 1%% edges removed)')
    parser.add_argument('--niden_decay_rate', type=float, default=0.00,
                        help='Rate at which removal rate changes per epoch')
    parser.add_argument('--niden_update_interval', type=int, default=5,
                        help='Interval (epochs) between NiDen graph updates')
    return parser.parse_args()


args = parse_args()
topks = eval(args.topks)
device = torch.device(f'cuda:{args.device:d}' if args.device!=-1 and torch.cuda.is_available() else 'cpu')

if args.seed != -1:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

if 'eig' not in args.model:
    args.eigs_dim = 0

print('Using',device)
print('Model Setting')
print(f'    hidden dim:{args.hidden_dim:d}')
print(f'    layers: {args.n_layers:d}')
print(f'    alpha: {args.alpha:f}')
print(f'    beta: {args.beta:f}')
print(f'    eigs dim: {args.eigs_dim:d}')
print(f'    sample hop: {args.sample_hop:d}')
print(f'model: {args.model:s}')

print('Train Setting')
print(f'    epochs: {args.epochs:d}')
print(f'    learning rate: {args.learning_rate:f}')

print('NiDen Setting')
print(f'    start_epoch: {args.niden_start_epoch:d}')
print(f'    rate_max: {args.niden_rate_max:.4f}')
print(f'    rate_min: {args.niden_rate_min:.4f}')
print(f'    decay_rate: {args.niden_decay_rate:.4f}')
print(f'    update_interval: {args.niden_update_interval:d}')

print('Data Setting')
train_file = os.path.join(args.data_dir, args.data, f'train.txt')
valid_file = os.path.join(args.data_dir, args.data, f'valid.txt')
test_file = os.path.join(args.data_dir, args.data, f'test.txt')
print(f'    train: {train_file:s}')
print(f'    valid: {valid_file:s}')
print(f'    test: {test_file:s}')
print(f'    offset: {args.offset:.1f}')

print('Test Setting')
print(f'    valid interval: {args.valid_interval:d}')
print(f'    test batch size: {args.test_batch_size:d}')
print(f'    stopping step: {args.stopping_step:d}')
print(f'    topks: ', topks)

print('---------------------------')
