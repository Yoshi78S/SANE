import torch
from torch.utils.data import Dataset
import pandas as pd
import parse
from parse import args
import scipy.sparse as sp
import torch.nn.functional as F
import numpy as np
import random
from torch import nn
import os
import math


class MyDataset(Dataset):
    def __init__(self, train_file, valid_file, test_file, device):
        self.device = device
        
        # train dataset
        train_data = pd.read_table(train_file, header=None, sep=' ')
        train_pos_data = train_data[train_data[2] >= args.offset]
        train_neg_data = train_data[train_data[2] < args.offset]
        self.train_data = torch.from_numpy(train_data.values).to(self.device)
        self.train_pos_user = torch.from_numpy(train_pos_data[0].values).to(self.device)
        self.train_pos_item = torch.from_numpy(train_pos_data[1].values).to(self.device)
        self.train_pos_unique_users = torch.unique(self.train_pos_user)
        self.train_pos_unique_items = torch.unique(self.train_pos_item)
        self.train_neg_user = torch.from_numpy(train_neg_data[0].values).to(self.device)
        self.train_neg_item = torch.from_numpy(train_neg_data[1].values).to(self.device)
        self.train_neg_unique_users = torch.unique(self.train_neg_user)
        self.train_neg_unique_items = torch.unique(self.train_neg_item)
        # valid dataset
        valid_data = pd.read_table(valid_file, header=None, sep=' ')
        valid_pos_data = valid_data[valid_data[2] >= args.offset]
        valid_neg_data = valid_data[valid_data[2] < args.offset]
        self.valid_data = torch.from_numpy(valid_data.values).to(self.device)
        self.valid_pos_user = torch.from_numpy(valid_pos_data[0].values).to(self.device)
        self.valid_pos_item = torch.from_numpy(valid_pos_data[1].values).to(self.device)
        self.valid_pos_unique_users = torch.unique(self.valid_pos_user)
        self.valid_pos_unique_items = torch.unique(self.valid_pos_item)
        self.valid_neg_user = torch.from_numpy(valid_neg_data[0].values).to(self.device)
        self.valid_neg_item = torch.from_numpy(valid_neg_data[1].values).to(self.device)
        self.valid_neg_unique_users = torch.unique(self.valid_neg_user)
        self.valid_neg_unique_items = torch.unique(self.valid_neg_item)
        # test dataset
        test_data = pd.read_table(test_file, header=None, sep=' ')
        test_pos_data = test_data[test_data[2] >= args.offset]
        test_neg_data = test_data[test_data[2] < args.offset]
        self.test_data = torch.from_numpy(test_data.values).to(self.device)
        self.test_pos_user = torch.from_numpy(test_pos_data[0].values).to(self.device)
        self.test_pos_item = torch.from_numpy(test_pos_data[1].values).to(self.device)
        self.test_pos_unique_users = torch.unique(self.test_pos_user)
        self.test_pos_unique_items = torch.unique(self.test_pos_item)
        self.test_neg_user = torch.from_numpy(test_neg_data[0].values).to(self.device)
        self.test_neg_item = torch.from_numpy(test_neg_data[1].values).to(self.device)
        self.test_neg_unique_users = torch.unique(self.test_neg_user)
        self.test_neg_unique_items = torch.unique(self.test_neg_item)
        self.num_users = max([self.train_pos_unique_users.max(),
                              self.train_neg_unique_users.max(),
                              self.valid_pos_unique_users.max(),
                              self.valid_neg_unique_users.max(),
                              self.test_pos_unique_users.max(),
                              self.test_neg_unique_users.max()]).cpu()+1
        self.num_items = max([self.train_pos_unique_items.max(),
                              self.train_neg_unique_items.max(),
                              self.valid_pos_unique_items.max(),
                              self.valid_neg_unique_items.max(),
                              self.test_pos_unique_items.max(),
                              self.test_neg_unique_items.max()]).cpu()+1
        self.num_nodes = self.num_users+self.num_items
        print('users: %d, items: %d.' % (self.num_users, self.num_items))
        print('train: %d pos + %d neg.' % (self.train_pos_user.shape[0], self.train_neg_user.shape[0]))
        print('valid: %d pos + %d neg.' % (self.valid_pos_user.shape[0], self.valid_neg_user.shape[0]))
        print('test: %d pos + %d neg.' % (self.test_pos_user.shape[0], self.test_neg_user.shape[0]))
        #
        self._train_neg_list = None
        self._train_pos_list = None
        self._valid_neg_list = None
        self._valid_pos_list = None
        self._test_neg_list = None
        self._test_pos_list = None
        self._A_pos = None
        self._A_neg = None
        self._degree_pos = None
        self._degree_neg = None
        self._tildeA = None
        self._tildeA_pos = None
        self._tildeA_neg = None
        self._indices = None
        self._paths = None
        self._values = None
        self._counts = None
        self._counts_sum = None
        self._L = None
        self._L_pos = None
        self._L_neg = None
        self._L_eigs = None
        self._norm_adj = None
        
        # NiDen: Store original edges for denoising
        self._original_train_pos_user = self.train_pos_user.clone()
        self._original_train_pos_item = self.train_pos_item.clone()
        self._original_train_neg_user = self.train_neg_user.clone()
        self._original_train_neg_item = self.train_neg_item.clone()
        
        # NiDen: Edge weights (1.0 = active, 0.0 = denoised/masked)
        # These weights are applied to adjacency matrix construction
        self._niden_pos_weights = torch.ones(self.train_pos_user.shape[0], device=self.device)
        self._niden_neg_weights = torch.ones(self.train_neg_user.shape[0], device=self.device)

    @property
    def train_pos_list(self):
        if self._train_pos_list is None:
            self._train_pos_list = [list(self.train_pos_item[self.train_pos_user == u].cpu().numpy()) for u in range(self.num_users)]
        return self._train_pos_list

    @ property
    def train_neg_list(self):
        if self._train_neg_list is None:
            self._train_neg_list = [list(self.train_neg_item[self.train_neg_user == u].cpu().numpy()) for u in range(self.num_users)]
        return self._train_neg_list

    @ property
    def valid_pos_list(self):
        if self._valid_pos_list is None:
            self._valid_pos_list = [list(self.valid_pos_item[self.valid_pos_user == u].cpu().numpy()) for u in self.valid_pos_unique_users]
        return self._valid_pos_list

    @ property
    def valid_neg_list(self):
        if self._valid_neg_list is None:
            self._valid_neg_list = [list(self.valid_neg_item[self.valid_neg_user == u].cpu().numpy()) for u in self.valid_pos_unique_users]
        return self._valid_neg_list

    @ property
    def test_pos_list(self):
        if self._test_pos_list is None:
            self._test_pos_list = [list(self.test_pos_item[self.test_pos_user == u].cpu().numpy()) for u in self.test_pos_unique_users]
        return self._test_pos_list

    @ property
    def test_neg_list(self):
        if self._test_neg_list is None:
            self._test_neg_list = [list(self.test_neg_item[self.test_neg_user == u].cpu().numpy()) for u in self.test_pos_unique_users]
        return self._test_neg_list

    @property
    def item_pos_pairs(self):
        if getattr(self, '_item_pos_pairs', None) is None:
            pos_i = []  # anchor items
            pos_j = []  # positive items
            
            # For each item, find users who gave positive feedback
            for item_id in range(self.num_items):
                # Users who gave positive feedback to this item
                pos_users = self.train_pos_user[self.train_pos_item == item_id].cpu().numpy()
                
                for u in pos_users:
                    # Other positive items of this user (excluding the anchor item)
                    user_pos_items = self.train_pos_item[self.train_pos_user == u].cpu().numpy()
                    other_pos_items = user_pos_items[user_pos_items != item_id]
                    
                    # Add pairs: (anchor_item, other_positive_item)
                    for other_item in other_pos_items:
                        pos_i.append(int(item_id))
                        pos_j.append(int(other_item))
            
            if len(pos_i) == 0:
                self._item_pos_pairs = (torch.empty(0, dtype=torch.long, device=parse.device), torch.empty(0, dtype=torch.long, device=parse.device))
            else:
                self._item_pos_pairs = (torch.tensor(pos_i, dtype=torch.long, device=parse.device), torch.tensor(pos_j, dtype=torch.long, device=parse.device))
        return self._item_pos_pairs

    @ property
    def item_neg_pairs(self):
        if getattr(self, '_item_neg_pairs', None) is None:
            neg_i = []  # anchor items
            neg_j = []  # negative items
            
            # For each item, create negative pairs
            for item_id in range(self.num_items):
                # Case 1: Users who gave positive feedback to this item
                pos_users = self.train_pos_user[self.train_pos_item == item_id].cpu().numpy()
                for u in pos_users:
                    # Negative items of this user
                    user_neg_items = self.train_neg_item[self.train_neg_user == u].cpu().numpy()
                    for neg_item in user_neg_items:
                        neg_i.append(int(item_id))  # anchor: positive item
                        neg_j.append(int(neg_item))  # negative: negative item
                
                # Case 2: Users who gave negative feedback to this item
                neg_users = self.train_neg_user[self.train_neg_item == item_id].cpu().numpy()
                for u in neg_users:
                    # Positive items of this user
                    user_pos_items = self.train_pos_item[self.train_pos_user == u].cpu().numpy()
                    for pos_item in user_pos_items:
                        neg_i.append(int(item_id))  # anchor: negative item
                        neg_j.append(int(pos_item))  # negative: positive item
            
            if len(neg_i) == 0:
                self._item_neg_pairs = (torch.empty(0, dtype=torch.long, device=parse.device), torch.empty(0, dtype=torch.long, device=parse.device))
            else:
                self._item_neg_pairs = (torch.tensor(neg_i, dtype=torch.long, device=parse.device), torch.tensor(neg_j, dtype=torch.long, device=parse.device))
        return self._item_neg_pairs

    @ property
    def A_pos(self):
        if self._A_pos is None:
            # Use NiDen weights (weight=0 means edge is masked/denoised)
            weights = self._niden_pos_weights
            # Create weighted adjacency matrix (both directions use same weight)
            indices = torch.cat([
                torch.stack([self.train_pos_user, self.train_pos_item+self.num_users]),
                torch.stack([self.train_pos_item+self.num_users, self.train_pos_user])], dim=1)
            values = torch.cat([weights, weights]).to(parse.device)
            self._A_pos = torch.sparse_coo_tensor(indices, values, torch.Size([self.num_nodes, self.num_nodes]))
        return self._A_pos

    @ property
    def degree_pos(self):
        if self._degree_pos is None:
            self._degree_pos = self.A_pos.sum(dim=1).to_dense()
        return self._degree_pos

    @ property
    def tildeA_pos(self):
        if self._tildeA_pos is None:
            D = self.degree_pos.float()
            D[D == 0.] = 1.
            D1 = torch.sparse_coo_tensor(
                torch.arange(self.num_nodes, device=parse.device).unsqueeze(0).repeat(2, 1),
                D**(-1/2),
                torch.Size([self.num_nodes, self.num_nodes]))
            D2 = torch.sparse_coo_tensor(
                torch.arange(self.num_nodes, device=parse.device).unsqueeze(0).repeat(2, 1),
                D**(-1/2),
                torch.Size([self.num_nodes, self.num_nodes]))
            self._tildeA_pos = torch.sparse.mm(torch.sparse.mm(D1, self.A_pos), D2)
        return self._tildeA_pos

    @ property
    def L_pos(self):
        if self._L_pos is None:
            D = torch.sparse_coo_tensor(
                torch.arange(self.num_nodes, device=parse.device).unsqueeze(0).repeat(2, 1),
                torch.ones(self.num_nodes, device=parse.device),
                torch.Size([self.num_nodes, self.num_nodes]))
            self._L_pos = D-self.tildeA_pos
        return self._L_pos

    @ property
    def A_neg(self):
        if self._A_neg is None:
            # Use NiDen weights (weight=0 means edge is masked/denoised)
            weights = self._niden_neg_weights
            indices = torch.cat([
                torch.stack([self.train_neg_user, self.train_neg_item + self.num_users]),
                torch.stack([self.train_neg_item + self.num_users, self.train_neg_user])], dim=1)
            values = torch.cat([weights, weights]).to(parse.device)
            self._A_neg = torch.sparse_coo_tensor(indices, values, torch.Size([self.num_nodes, self.num_nodes]))
        return self._A_neg

    @ property
    def degree_neg(self):
        if self._degree_neg is None:
            self._degree_neg = self.A_neg.sum(dim=1).to_dense()
        return self._degree_neg

    @ property
    def tildeA_neg(self):
        if self._tildeA_neg is None:
            D = self.degree_neg.float()
            D[D == 0.] = 1.
            D1 = torch.sparse_coo_tensor(
                torch.arange(self.num_nodes, device=parse.device).unsqueeze(0).repeat(2, 1),
                D**(-1/2),
                torch.Size([self.num_nodes, self.num_nodes]))
            D2 = torch.sparse_coo_tensor(
                torch.arange(self.num_nodes, device=parse.device).unsqueeze(0).repeat(2, 1),
                D**(-1/2),
                torch.Size([self.num_nodes, self.num_nodes]))
            self._tildeA_neg = torch.sparse.mm(torch.sparse.mm(D1, self.A_neg), D2)
        return self._tildeA_neg

    @ property
    def L_neg(self):
        if self._L_neg is None:
            D = torch.sparse_coo_tensor(
                torch.arange(self.num_nodes, device=parse.device).unsqueeze(0).repeat(2, 1),
                torch.ones(self.num_nodes, device=parse.device),
                torch.Size([self.num_nodes, self.num_nodes]))
            self._L_neg = D-self.tildeA_neg
        return self._L_neg

    @ property
    def L(self):
        if self._L is None:
            self._L = (self.L_pos+args.alpha*self.L_neg)/(1+args.alpha)
        return self._L

    @ property
    def L_eigs(self):
        if self._L_eigs is None:
            if args.eigs_dim == 0:
                self._L_eigs = torch.tensor([]).to(parse.device)
            else:
                _, self._L_eigs = sp.linalg.eigs(
                    sp.csr_matrix(
                        (self.L._values().cpu(), self.L._indices().cpu()),
                        (self.num_nodes, self.num_nodes)),
                    k=args.eigs_dim,
                    which='SR')
                self._L_eigs = torch.tensor(self._L_eigs.real).to(parse.device)
                self._L_eigs = F.layer_norm(self._L_eigs, normalized_shape=(args.eigs_dim,))
        return self._L_eigs

    @ property
    def norm_adj(self):
        if self._norm_adj is None:
            # Create normalized adjacency matrix combining positive and negative interactions
            # Similar to how L is computed: (tildeA_pos + alpha * tildeA_neg) / (1 + alpha)
            self._norm_adj = (self.tildeA_pos + args.alpha * self.tildeA_neg) / (1 + args.alpha)
        return self._norm_adj

    def sample(self):
        if self._indices is None:
            self._indices = torch.cat([
                torch.stack([self.train_pos_user, self.train_pos_item+self.num_users]),
                torch.stack([self.train_pos_item+self.num_users, self.train_pos_user]),
                torch.stack([self.train_neg_user, self.train_neg_item+self.num_users]),
                torch.stack([self.train_neg_item+self.num_users, self.train_neg_user])], dim=1)
            self._paths = torch.cat([
                torch.ones(self.train_pos_user.shape).repeat(2),
                torch.zeros(self.train_neg_user.shape).repeat(2)], dim=0).long().to(parse.device)
            sorted_indices = torch.argsort(self._indices[0, :])
            self._indices = self._indices[:, sorted_indices]
            self._paths = self._paths[sorted_indices]
            self._counts = torch.bincount(self._indices[0], minlength=self.num_nodes)
            self._counts_sum = torch.cumsum(self._counts, dim=0)
            d = torch.sqrt(self._counts)
            d[d == 0.] = 1.
            d = 1./d
            self._values = torch.ones(self._indices.shape[1]).to(
                parse.device)*d[self._indices[0]]*d[self._indices[1]]
        res_X, res_Y = [], []
        record_X = []
        X,  Y,  = self._indices,  torch.ones_like(self._paths).long()*2+self._paths
        loop_indices = torch.zeros_like(Y).bool()
        for hop in range(args.sample_hop):
            loop_indices = loop_indices | (X[0] == X[1])
            for i in range(hop % 2, hop, 2):
                loop_indices = loop_indices | (record_X[i][1] == X[1])
            record_X.append(X)
            res_X.append(X[:, ~loop_indices])
            res_Y.append(Y[~loop_indices]-2)
            next_indices = self._counts_sum[X[1]]-(torch.rand(X.shape[1]).to(parse.device)*self._counts[X[1]]).long()-1
            X = torch.stack([X[0], self._indices[1, next_indices]], dim=0)
            Y = Y*2+self._paths[next_indices]
        return res_X, res_Y

    def sample_item_pairs(self, sample_size: int = None):
        """Randomly sample item pos-pos and pos-neg pairs for an epoch.
        Returns tuple of tensors: (pos_i, pos_j, neg_i, neg_j) on parse.device.
        If sample_size is None, use args.item_pair_sample_size.
        """

        pos_lists = self.train_pos_list
        neg_lists = self.train_neg_list

        pos_i = []
        pos_j = []
        neg_i = []
        neg_j = []

        num_users = int(self.num_users)
        # Limit trials to avoid infinite loops
        max_trials = max(1000000, sample_size * 10)
        trials = 0
        while (len(pos_i) < sample_size or len(neg_i) < sample_size) and trials < max_trials:
            trials += 1
            u = random.randrange(0, num_users)
            p_list = pos_lists[u]
            n_list = neg_lists[u]
            # pos-pos
            if len(p_list) >= 2 and len(pos_i) < sample_size:
                a, b = np.random.choice(len(p_list), size=2, replace=False)
                pos_i.append(int(p_list[a]))
                pos_j.append(int(p_list[b]))
            # pos-neg
            if len(p_list) >= 1 and len(n_list) >= 1 and len(neg_i) < sample_size:
                a = np.random.randint(0, len(p_list))
                b = np.random.randint(0, len(n_list))
                neg_i.append(int(p_list[a]))
                neg_j.append(int(n_list[b]))

        pos_i_t = torch.tensor(pos_i, dtype=torch.long, device=parse.device) if len(pos_i) > 0 else torch.empty(0, dtype=torch.long, device=parse.device)
        pos_j_t = torch.tensor(pos_j, dtype=torch.long, device=parse.device) if len(pos_j) > 0 else torch.empty(0, dtype=torch.long, device=parse.device)
        neg_i_t = torch.tensor(neg_i, dtype=torch.long, device=parse.device) if len(neg_i) > 0 else torch.empty(0, dtype=torch.long, device=parse.device)
        neg_j_t = torch.tensor(neg_j, dtype=torch.long, device=parse.device) if len(neg_j) > 0 else torch.empty(0, dtype=torch.long, device=parse.device)

        return pos_i_t, pos_j_t, neg_i_t, neg_j_t

    def compute_dynamic_thresholds(self, niden_epoch, pos_nsim, neg_nsim):
        """Compute dynamic thresholds for NiDen based on Nsim percentiles.
        
        For positive edges: remove edges with Nsim below (rate_pos)-th percentile
            - rate_pos increases over time (relaxed -> strict)
            - rate_pos = min(rate_min + decay, rate_max)
            
        For negative edges: remove edges with Nsim above (1 - rate_neg)-th percentile
            - rate_neg increases over time (relaxed -> strict)
            - rate_neg = min(rate_min + decay, rate_max)
        
        Args:
            niden_epoch: Epoch count since NiDen started (epoch - start_denoise)
            pos_nsim: Nsim values for positive edges [num_pos_edges]
            neg_nsim: Nsim values for negative edges [num_neg_edges]
        
        Returns:
            lambda_pos: Threshold for positive edges (Nsim value at percentile)
            lambda_neg: Threshold for negative edges (Nsim value at percentile)
        """
        # Compute decay factor (actually an increase factor)
        decay = args.niden_decay_rate * niden_epoch
        
        # rate_pos: starts at rate_min, increases to rate_max over time
        # (initially relaxed/few removals, becomes strict/more removals)
        rate_pos = min(args.niden_rate_min + decay, args.niden_rate_max)
        
        # rate_neg: starts at rate_min, increases to rate_max over time
        # (initially relaxed/few removals, becomes strict/more removals)
        rate_neg = min(args.niden_rate_min + decay, args.niden_rate_max)
        
        # Compute lambda_pos: (rate_pos)-th percentile of pos_nsim
        # Edges with Nsim < lambda_pos will be masked
        # If rate_pos == 0, no edges should be masked, so set lambda_pos = 0
        if rate_pos == 0 or pos_nsim.numel() == 0:
            lambda_pos = 0.0
        else:
            lambda_pos = torch.quantile(pos_nsim.float(), rate_pos).item()
        
        # Compute lambda_neg: (1 - rate_neg)-th percentile of neg_nsim
        # Edges with Nsim >= lambda_neg will be masked
        # If rate_neg == 0, no edges should be masked, so set lambda_neg > max possible (1.0+)
        if rate_neg == 0 or neg_nsim.numel() == 0:
            lambda_neg = 1.1  # > 1.0, so no edges will be masked
        else:
            lambda_neg = torch.quantile(neg_nsim.float(), 1.0 - rate_neg).item()
        
        return lambda_pos, lambda_neg, rate_pos, rate_neg

    def update_graph_niden(self, pos_nsim, neg_nsim, lambda_pos, lambda_neg):
        """Update edge weights based on NiDen denoising (like official implementation).
        
        Instead of removing edges, we set their weights to 0.
        - Positive edges with Nsim < lambda_pos: weight = 0 (noisy)
        - Negative edges with Nsim >= lambda_neg: weight = 0 (likely not true dislikes)
        
        Uses ORIGINAL edges and updates weights each time (not cumulative deletion).
        
        Args:
            pos_nsim: Nsim values for ALL original positive edges [num_pos_edges]
            neg_nsim: Nsim values for ALL original negative edges [num_neg_edges]
            lambda_pos: Threshold for positive edge filtering
            lambda_neg: Threshold for negative edge filtering
        
        Returns:
            dict with statistics about masked edges
        """
        # Detach
        pos_nsim = pos_nsim.detach()
        neg_nsim = neg_nsim.detach()
        
        # Graph weights: binary mask only (1 = active, 0 = masked)
        # Nsim-based weighting is done in BPR loss, not in graph structure
        
        # Positive edges: mask if Nsim < lambda_pos
        pos_mask = pos_nsim >= lambda_pos
        new_pos_weights = pos_mask.float()  # Binary: 1 or 0
        num_pos_masked = (~pos_mask).sum().item()
        
        # Negative edges: mask if Nsim >= lambda_neg (high similarity = likely not true dislike)
        if neg_nsim.numel() > 0:
            neg_mask = neg_nsim < lambda_neg
            new_neg_weights = neg_mask.float()  # Binary: 1 or 0
            num_neg_masked = (~neg_mask).sum().item()
        else:
            new_neg_weights = torch.tensor([], device=self.device)
            num_neg_masked = 0
        
        # Update weights
        self._niden_pos_weights = new_pos_weights
        self._niden_neg_weights = new_neg_weights
        
        # Clear cached adjacency matrices to rebuild with new weights
        self._invalidate_graph_cache()
        
        # Count active edges (weight > 0)
        active_pos = (new_pos_weights > 0).sum().item()
        active_neg = (new_neg_weights > 0).sum().item() if new_neg_weights.numel() > 0 else 0
        
        return {
            'pos_masked': num_pos_masked,
            'neg_masked': num_neg_masked,
            'lambda_pos': lambda_pos,
            'lambda_neg': lambda_neg,
            'active_pos_edges': active_pos,
            'active_neg_edges': active_neg,
            'total_pos_edges': len(self.train_pos_user),
            'total_neg_edges': len(self.train_neg_user)
        }
    
    def _invalidate_graph_cache(self):
        """Invalidate all cached graph structures after NiDen update."""
        self._train_pos_list = None
        self._train_neg_list = None
        self._A_pos = None
        self._A_neg = None
        self._degree_pos = None
        self._degree_neg = None
        self._tildeA_pos = None
        self._tildeA_neg = None
        self._L_pos = None
        self._L_neg = None
        self._L = None
        self._L_eigs = None
        self._norm_adj = None
        self._indices = None
        self._paths = None
        self._values = None
        self._counts = None
        self._counts_sum = None
        self._item_pos_pairs = None
        self._item_neg_pairs = None

    def reset_graph_to_original(self):
        """Reset edge weights to original state (all weights = 1.0)."""
        self._niden_pos_weights = torch.ones(self.train_pos_user.shape[0], device=self.device)
        self._niden_neg_weights = torch.ones(self.train_neg_user.shape[0], device=self.device)
        
        self._invalidate_graph_cache()


# Create dataset instance
dataset = MyDataset(
    parse.train_file, 
    parse.valid_file,
    parse.test_file, 
    parse.device
)

if __name__ == "__main__":
    # ユーザIDを指定（ここではユーザ0）して、そのポジティブ／ネガティブリストを表示する
    u = 15248
    pos = dataset.train_pos_list[u]
    neg = dataset.train_neg_list[u]
    print(len(dataset.train_pos_list))
    print(f'ユーザ {u} の train positive items (count={len(pos)}):', pos)
    print(f'ユーザ {u} の train negative items (count={len(neg)}):', neg)