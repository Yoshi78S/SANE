import parse
from parse import args
from dataloader import dataset
from model import Model
import torch
import sys
import os
from datetime import datetime


class Tee:
    def __init__(self, file):
        self.file = file
        self.stdout = sys.stdout

    def write(self, data):
        self.file.write(data)
        self.file.flush()  # 即座にファイルに書き込む
        self.stdout.write(data)
        self.stdout.flush()

    def flush(self):
        self.file.flush()
        self.stdout.flush()


def print_test_result():
    global best_epoch, test_pre, test_recall, test_ndcg, test_neg_pro
    print(f'Test Result(at {best_epoch:d} epoch):')
    for i, k in enumerate(parse.topks):
        print(f'ndcg@{k:d} = {test_ndcg[i]:f}, recall@{k:d} = {test_recall[i]:f}, pre@{k:d} = {test_pre[i]:f}, neg_pro@{k:d} = {test_neg_pro[i]:f}.')


def niden_update(model, epoch):
    """Perform NiDen graph denoising update.
    
    Args:
        model: The model instance
        epoch: Current epoch number
    
    Returns:
        dict with update statistics, or None if no update performed
    """
    niden_epoch = epoch - args.niden_start_epoch
    
    # Compute Nsim for all edges
    model.eval()
    with torch.no_grad():
        pos_nsim, neg_nsim = model.compute_all_nsim()
    
    # Compute dynamic thresholds based on Nsim percentiles
    lambda_pos, lambda_neg, rate_pos, rate_neg = dataset.compute_dynamic_thresholds(
        niden_epoch, pos_nsim, neg_nsim
    )
    
    # Debug: Print Nsim distribution (only first few NiDen epochs)
    if niden_epoch <= 3:
        print(f'[NiDen Debug] pos_nsim: min={pos_nsim.min():.4f}, max={pos_nsim.max():.4f}, mean={pos_nsim.mean():.4f}')
        if len(neg_nsim) > 0:
            print(f'[NiDen Debug] neg_nsim: min={neg_nsim.min():.4f}, max={neg_nsim.max():.4f}, mean={neg_nsim.mean():.4f}')
    
    # Update graph structure
    stats = dataset.update_graph_niden(pos_nsim, neg_nsim, lambda_pos, lambda_neg)
    stats['rate_pos'] = rate_pos
    stats['rate_neg'] = rate_neg
    
    return stats


def train():
    train_loss, train_loss_bpr, train_loss_cl = model.train_func()
    if epoch % args.show_loss_interval == 0:
        print(f'epoch {epoch:d}, train_loss = {train_loss:f}, train_loss_bpr = {train_loss_bpr:f}, train_loss_cl = {train_loss_cl:f}')


def valid(epoch):
    global best_valid_ndcg, best_epoch, test_pre, test_recall, test_ndcg, test_neg_pro
    valid_pre, valid_recall, valid_ndcg, valid_neg_pro = model.valid_func()
    for i, k in enumerate(parse.topks):
        print(f'[{epoch:d}/{args.epochs:d}] Valid Result: ndcg@{k:d} = {valid_ndcg[i]:f}, recall@{k:d} = {valid_recall[i]:f}, pre@{k:d} = {valid_pre[i]:f}, neg_pro@{k:d} = {valid_neg_pro[i]:f}.')
    if valid_ndcg[-1] > best_valid_ndcg:
        best_valid_ndcg, best_epoch = valid_ndcg[-1], epoch
        test_pre, test_recall, test_ndcg, test_neg_pro = model.test_func()
        print_test_result()
        return True
    return False


# Setup output logging
os.makedirs('results', exist_ok=True)
log_file = f'results/{args.data}_niden_rmax{args.niden_rate_max}_rmin{args.niden_rate_min}_decay{args.niden_decay_rate}_start{args.niden_start_epoch}.txt'

if __name__ == "__main__":
    with open(log_file, 'w', buffering=1) as log_f:  # Line buffering
        original_stdout = sys.stdout
        sys.stdout = Tee(log_f)

        print(f"Training started at: {datetime.now()}")
        print(f"Dataset: {args.data}")
        print(f"Arguments: {vars(args)}")
        print("-" * 50)

        model = Model(dataset).to(parse.device)

        best_valid_ndcg, best_epoch = 0., 0
        test_pre, test_recall, test_ndcg, test_neg_pro = torch.zeros(len(args.topks)), torch.zeros(len(args.topks)), torch.zeros(len(args.topks)), torch.zeros(len(args.topks))
        valid(epoch=0)
        for epoch in range(1, args.epochs+1):
            # Update model's current epoch for NiDen weighting
            model.current_epoch = epoch
            
            # NiDen: Perform graph denoising update
            if epoch > args.niden_start_epoch and (epoch - args.niden_start_epoch) % args.niden_update_interval == 0:
                stats = niden_update(model, epoch)
                print(f'[NiDen] epoch {epoch}: pos_masked={stats["pos_masked"]}, neg_masked={stats["neg_masked"]}, '
                      f'rate_pos={stats["rate_pos"]:.4f}, rate_neg={stats["rate_neg"]:.4f}, '
                      f'λ_pos={stats["lambda_pos"]:.4f}, λ_neg={stats["lambda_neg"]:.4f}, '
                      f'active: pos={stats["active_pos_edges"]}/{stats["total_pos_edges"]}, neg={stats["active_neg_edges"]}/{stats["total_neg_edges"]}')
            
            # Per-epoch sample item and user pairs to limit computation
            try:
                # Sample item pairs
                pos_i_t, pos_j_t, neg_i_t, neg_j_t = dataset.sample_item_pairs(sample_size=args.item_pair_sample_size)
                dataset._epoch_item_pairs = (pos_i_t, pos_j_t, neg_i_t, neg_j_t)

            except Exception:
                dataset._epoch_item_pairs = None
                dataset._epoch_user_pairs = None
            train()

            if epoch % args.valid_interval == 0:
                if not valid(epoch) and epoch-best_epoch >= args.stopping_step*args.valid_interval:
                    break
        print('---------------------------')
        print_test_result()
        print(f"Training completed at: {datetime.now()}")
        print(f"Results saved to: {log_file}")

        # Ensure all data is written
        sys.stdout.flush()
        log_f.flush()
