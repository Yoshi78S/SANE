import parse
from parse import args
import torchsparsegradutils
import utils
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import structured_negative_sampling


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.lambda0 = nn.Parameter(torch.zeros(1))
        self.path_emb = nn.Embedding(2**(args.sample_hop+1)-2, 1)
        nn.init.zeros_(self.path_emb.weight)
        self.sqrt_dim = 1./torch.sqrt(torch.tensor(args.hidden_dim))
        self.sqrt_eig = 1./torch.sqrt(torch.tensor(args.eigs_dim))
        self.my_parameters = [
            {'params': self.lambda0, 'weight_decay': 1e-2},
            {'params': self.path_emb.parameters()},
        ]

    def forward(self, q, k, v,  indices, eigs, path_type):
        ni, nx, ny, nz = [], [], [], []
        for i, pt in zip(indices, path_type):
            x = torch.mul(q[i[0]], k[i[1]]).sum(dim=-1)*self.sqrt_dim
            nx.append(x)
            if 'eig' in args.model:
                if args.eigs_dim == 0:
                    y = torch.zeros(i.shape[1]).to(parse.device)
                else:
                    y = torch.mul(eigs[i[0]], eigs[i[1]]).sum(dim=-1)
                ny.append(y)
            if 'path' in args.model:
                z = self.path_emb(pt).view(-1)
                nz.append(z)
            ni.append(i)
        i = torch.concat(ni, dim=-1)
        s = []
        s.append(torch.concat(nx, dim=-1))
        if 'eig' in args.model:
            s[0] = s[0]+torch.exp(self.lambda0)*torch.concat(ny, dim=-1)
        if 'path' in args.model:
            s.append(torch.concat(nz, dim=-1))
        s = [utils.sparse_softmax(i, _, q.shape[0]) for _ in s]
        s = torch.stack(s, dim=1).mean(dim=1)
        return torchsparsegradutils.sparse_mm(torch.sparse_coo_tensor(i, s, torch.Size([q.shape[0], k.shape[0]])), v)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.self_attention = Attention()
        self.my_parameters = self.self_attention.my_parameters

    def forward(self, x, indices, eigs, path_type):
        y = F.layer_norm(x, normalized_shape=(args.hidden_dim,))
        y = self.self_attention(
            y, y, y,
            indices,
            eigs,
            path_type)
        return y


class Model(nn.Module):
    def __init__(self, dataset):
        super(Model, self).__init__()
        self.dataset = dataset
        self.hidden_dim = args.hidden_dim
        self.n_layers = args.n_layers
        self.embedding_user = nn.Embedding(self.dataset.num_users, self.hidden_dim)
        self.embedding_item = nn.Embedding(self.dataset.num_items, self.hidden_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        self.my_parameters = [
            {'params': self.embedding_user.parameters()},
            {'params': self.embedding_item.parameters()},
        ]
        self.layers = []
        for i in range(args.n_layers):
            layer = Encoder().to(parse.device)
            self.layers.append(layer)
            self.my_parameters.extend(layer.my_parameters)
        self._users, self._items = None, None
        self._embeddings_list = None  # Store per-layer embeddings for NiDen
        self.current_epoch = 0  # Track current epoch for NiDen
        self.optimizer = torch.optim.Adam(
            self.my_parameters,
            lr=args.learning_rate)

    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        for i in range(self.n_layers):
            indices, paths = self.dataset.sample()
            all_emb = self.layers[i](all_emb,
                                     indices,
                                     self.dataset.L_eigs,
                                     paths)
            embs.append(all_emb)
        # Store per-layer embeddings for NiDen Nsim computation
        self._embeddings_list = embs
        embs_stacked = torch.stack(embs, dim=1)
        light_out = torch.mean(embs_stacked, dim=1)
        self._users, self._items = torch.split(light_out, [self.dataset.num_users, self.dataset.num_items])

    def compute_nsim_for_edges(self, user_indices, item_indices, use_layers=(2, 3)):
        """Compute Nsim (neighborhood similarity) for given user-item edges.
        
        Uses staggered embeddings from specified layers.
        Nsim = 0.5 * (e1_u · e2_i + e2_u · e1_i)
        Then scaled from [-1, 1] to [0, 1]
        
        Args:
            user_indices: User indices of edges [num_edges]
            item_indices: Item indices of edges [num_edges]
            use_layers: Tuple of (layer1, layer2) indices for staggered embeddings
        
        Returns:
            nsim: Nsim values for each edge [num_edges], scaled to [0, 1]
        """
        if self._embeddings_list is None:
            self.computer()
        
        # Adjust layer indices if model has fewer layers
        layer1 = min(use_layers[0], len(self._embeddings_list) - 1)
        layer2 = min(use_layers[1], len(self._embeddings_list) - 1)
        
        # Get embeddings from specified layers
        emb1 = self._embeddings_list[layer1]  # [num_users + num_items, hidden_dim]
        emb2 = self._embeddings_list[layer2]  # [num_users + num_items, hidden_dim]
        
        num_users = self.dataset.num_users
        
        # Split into user and item embeddings
        e1_users, e1_items = torch.split(emb1, [num_users, self.dataset.num_items])
        e2_users, e2_items = torch.split(emb2, [num_users, self.dataset.num_items])
        
        # Get embeddings for the given edges
        e1_u = F.normalize(e1_users[user_indices], dim=-1)  # [num_edges, hidden_dim]
        e2_i = F.normalize(e2_items[item_indices], dim=-1)  # [num_edges, hidden_dim]
        e2_u = F.normalize(e2_users[user_indices], dim=-1)  # [num_edges, hidden_dim]
        e1_i = F.normalize(e1_items[item_indices], dim=-1)  # [num_edges, hidden_dim]
        
        # Compute Nsim = 0.5 * (e1_u · e2_i + e2_u · e1_i)
        sim1 = torch.sum(e1_u * e2_i, dim=-1)  # [num_edges]
        sim2 = torch.sum(e2_u * e1_i, dim=-1)  # [num_edges]
        nsim = 0.5 * (sim1 + sim2)
        
        # Scale from [-1, 1] to [0, 1]
        nsim = (nsim + 1) / 2
        
        return nsim

    def compute_all_nsim(self):
        """Compute Nsim for all positive and negative edges.
        
        Uses CURRENT edges (for cumulative NiDen updates).
        
        Returns:
            pos_nsim: Nsim values for current positive edges
            neg_nsim: Nsim values for current negative edges
        """
        with torch.no_grad():
            # Ensure embeddings are computed
            if self._embeddings_list is None:
                self.computer()
            
            # Get CURRENT edge indices (not original)
            pos_users = self.dataset.train_pos_user
            pos_items = self.dataset.train_pos_item
            neg_users = self.dataset.train_neg_user
            neg_items = self.dataset.train_neg_item
            
            # Compute Nsim for all edges using layers 2 and 3 (like NiDen paper)
            pos_nsim = self.compute_nsim_for_edges(pos_users, pos_items, use_layers=(2, 3))
            if len(neg_users) > 0:
                neg_nsim = self.compute_nsim_for_edges(neg_users, neg_items, use_layers=(2, 3))
            else:
                neg_nsim = torch.tensor([], device=pos_nsim.device)
            
            return pos_nsim, neg_nsim

    def compute_batch_nsim(self, user_indices, item_indices, use_layers=(1, 0)):
        """Compute Nsim for a batch of edges (used for sample re-weighting).
        
        Uses layers 1 and 0 (like NiDen's get_batch_weight: embedding_list[1] and [0]).
        
        Args:
            user_indices: User indices [batch_size]
            item_indices: Item indices [batch_size]
            use_layers: Tuple of layer indices to use
        
        Returns:
            nsim: Nsim values [batch_size]
        """
        with torch.no_grad():
            return self.compute_nsim_for_edges(user_indices, item_indices, use_layers=use_layers)

    def evaluate(self, test_pos_unique_users, test_pos_list, test_neg_list):
        self.eval()
        if self._users is None:
            self.computer()
        user_emb, item_emb = self._users, self._items
        max_K = max(parse.topks)
        all_pre = torch.zeros(len(parse.topks))
        all_recall = torch.zeros(len(parse.topks))
        all_ndcg = torch.zeros(len(parse.topks))
        all_neg = torch.zeros(len(parse.topks))
        with torch.no_grad():
            users = test_pos_unique_users
            for i in range(0, users.shape[0], args.test_batch_size):
                batch_users = users[i:i+args.test_batch_size]
                user_e = user_emb[batch_users]
                rating = torch.mm(user_e, item_emb.t())
                for j, u in enumerate(batch_users):
                    rating[j, self.dataset.train_pos_list[u]] = -(1 << 10)
                    rating[j, self.dataset.train_neg_list[u]] = -(1 << 10)
                _, rating = torch.topk(rating, k=max_K)
                # existing metrics
                pre, recall, ndcg = utils.test(
                    rating,
                    test_pos_list[i:i+args.test_batch_size])
                all_pre += pre
                all_recall += recall
                all_ndcg += ndcg
                # compute proportion of recommended items that are in provided negative lists
                # convert predictions to numpy for per-row membership tests
                rating_np = rating.cpu().numpy()
                neg_rows = []
                batch_neg_lists = test_neg_list[i:i+args.test_batch_size]
                for row_idx in range(rating_np.shape[0]):
                    pred_row = rating_np[row_idx]
                    neg_set = set(batch_neg_lists[row_idx]) if batch_neg_lists[row_idx] is not None else set()
                    neg_rows.append([1 if int(x) in neg_set else 0 for x in pred_row])
                if len(neg_rows) > 0:
                    neg_arr = np.array(neg_rows)
                    for idx_k, k in enumerate(parse.topks):
                        now_k = min(k, neg_arr.shape[1])
                        neg_count_per_user = neg_arr[:, :now_k].sum(axis=1)
                        # sum proportion per user (neg_count / k) then sum across users
                        all_neg[idx_k] += neg_count_per_user.sum() / float(now_k)
            all_pre /= users.shape[0]
            all_recall /= users.shape[0]
            all_ndcg /= users.shape[0]
            all_neg /= users.shape[0]
        return all_pre, all_recall, all_ndcg, all_neg

    def valid_func(self):
        return self.evaluate(self.dataset.valid_pos_unique_users, self.dataset.valid_pos_list, self.dataset.valid_neg_list)

    def test_func(self):
        return self.evaluate(self.dataset.test_pos_unique_users, self.dataset.test_pos_list, self.dataset.test_neg_list)

    def train_func(self):
        self.train()
        pos_u = self.dataset.train_pos_user
        pos_i = self.dataset.train_pos_item
        indices = torch.randperm(self.dataset.train_neg_user.shape[0])
        neg_u = self.dataset.train_neg_user[indices]
        neg_i = self.dataset.train_neg_item[indices]
        all_j = structured_negative_sampling(
                torch.concat([torch.stack([pos_u, pos_i]), torch.stack([neg_u, neg_i])], dim=1),
                num_nodes=self.dataset.num_items)[2]
        pos_j, neg_j = torch.split(all_j, [pos_u.shape[0], neg_u.shape[0]])
        loss, loss_bpr, loss_item = self.loss_one_batch(pos_u, pos_i, pos_j, neg_u, neg_i, neg_j)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss, loss_bpr, loss_item

    def loss_one_batch(self, pos_u, pos_i, pos_j, neg_u, neg_i, neg_j):
        self.computer()
        all_user, all_item = self._users, self._items
        pos_u_emb0, pos_u_emb = self.embedding_user(pos_u), all_user[pos_u]
        pos_i_emb0, pos_i_emb = self.embedding_item(pos_i), all_item[pos_i]
        pos_j_emb0, pos_j_emb = self.embedding_item(pos_j), all_item[pos_j]
        neg_u_emb0, neg_u_emb = self.embedding_user(neg_u), all_user[neg_u]
        neg_i_emb0, neg_i_emb = self.embedding_item(neg_i), all_item[neg_i]
        neg_j_emb0, neg_j_emb = self.embedding_item(neg_j), all_item[neg_j]
        pos_scores_ui = torch.sum(torch.mul(pos_u_emb, pos_i_emb), dim=-1)
        pos_scores_uj = torch.sum(torch.mul(pos_u_emb, pos_j_emb), dim=-1)
        neg_scores_ui = torch.sum(torch.mul(neg_u_emb, neg_i_emb), dim=-1)
        neg_scores_uj = torch.sum(torch.mul(neg_u_emb, neg_j_emb), dim=-1)
        
        # Regularization loss (same as original SIGformer)
        if neg_u.shape[0] > 0:
            reg_loss = (1/2)*(pos_u_emb0.norm(2).pow(2) +
                              pos_i_emb0.norm(2).pow(2) +
                              pos_j_emb0.norm(2).pow(2) +
                              neg_u_emb0.norm(2).pow(2) +
                              neg_i_emb0.norm(2).pow(2) +
                              neg_j_emb0.norm(2).pow(2))/float(pos_u.shape[0]+neg_u.shape[0])
        else:
            reg_loss = (1/2)*(pos_u_emb0.norm(2).pow(2) +
                              pos_i_emb0.norm(2).pow(2) +
                              pos_j_emb0.norm(2).pow(2))/float(pos_u.shape[0])
        
        # BPR Loss with NiDen Nsim weighting
        # After niden_start_epoch, weight samples by Nsim
        # - Positive: weight = Nsim / max(Nsim) (high Nsim = confident)
        # - Negative: weight = beta * (1 - Nsim) / max(1 - Nsim) (low Nsim = strong negative)
        
        pos_scores_diff = pos_scores_uj - pos_scores_ui
        
        use_niden_weighting = self.current_epoch > args.niden_start_epoch
        
        if use_niden_weighting:
            # Compute Nsim for positive edges
            with torch.no_grad():
                pos_nsim = self.compute_batch_nsim(pos_u, pos_i, use_layers=(1, 0))
                # Normalize: weight = Nsim / max(Nsim)
                pos_max = pos_nsim.max()
                if pos_max > 0:
                    pos_weights = pos_nsim / pos_max
                else:
                    pos_weights = torch.ones_like(pos_nsim)
            
            # Weighted positive loss
            pos_loss = F.softplus(pos_scores_diff) * pos_weights
            
            if neg_u.shape[0] > 0:
                neg_scores_diff = neg_scores_uj - neg_scores_ui
                with torch.no_grad():
                    neg_nsim = self.compute_batch_nsim(neg_u, neg_i, use_layers=(1, 0))
                    # Negative: weight = (1 - Nsim) / max(1 - Nsim), then multiply by beta
                    neg_strength = 1.0 - neg_nsim
                    neg_max = neg_strength.max()
                    if neg_max > 0:
                        neg_weights = args.beta * (neg_strength / neg_max)
                    else:
                        neg_weights = args.beta * torch.ones_like(neg_nsim)
                
                # Weighted negative loss
                neg_loss = F.softplus(neg_scores_diff) * neg_weights
                
                # Combine and average
                all_losses = torch.cat([pos_loss, neg_loss])
                loss_bpr = all_losses.mean()
            else:
                loss_bpr = pos_loss.mean()
        else:
            # Original SIGformer BPR loss (before NiDen starts)
            if neg_u.shape[0] > 0:
                neg_scores_diff = neg_scores_uj - neg_scores_ui
                scores = torch.cat([pos_scores_diff, args.beta * neg_scores_diff])
            else:
                scores = pos_scores_diff
            loss_bpr = torch.mean(F.softplus(scores))
        # --- Item-pair contrastive loss (2-hop via users) ---
        # Ensure loss_item is always defined
        loss_item = torch.tensor(0., device=all_item.device)
        # Prefer per-epoch sampled pairs if available
        epoch_pairs = getattr(self.dataset, '_epoch_item_pairs', None)
        if epoch_pairs is not None:
            try:
                item_pos_i, item_pos_j, item_neg_i, item_neg_j = epoch_pairs
            except Exception:
                item_pos_i = item_pos_j = item_neg_i = item_neg_j = None
        else:
            try:
                item_pos_i, item_pos_j = self.dataset.item_pos_pairs
                item_neg_i, item_neg_j = self.dataset.item_neg_pairs
            except Exception:
                item_pos_i = item_pos_j = item_neg_i = item_neg_j = None

        if item_pos_i is not None and item_pos_i.numel() > 0:
            # Use cosine similarity (normalize embeddings)
            item_emb_norm = F.normalize(all_item, dim=1)
            tau = max(args.item_pair_tau, 1e-6)

            if item_neg_i is None or item_neg_i.numel() == 0:
                # No negatives -> loss 0
                loss_item = torch.tensor(0., device=item_pos_i.device)
            else:
                # Memory-efficient InfoNCE loss
                # Limit the number of anchors and negatives to avoid OOM
                max_anchors = min(10000, item_pos_i.shape[0])  # Limit anchors
                max_negatives = min(10000, item_neg_i.shape[0])  # Limit negatives

                # Sample if too many
                if item_pos_i.shape[0] > max_anchors:
                    anchor_indices = torch.randperm(item_pos_i.shape[0])[:max_anchors]
                    anchor_pos_i = item_pos_i[anchor_indices]
                    anchor_pos_j = item_pos_j[anchor_indices]
                else:
                    anchor_pos_i = item_pos_i
                    anchor_pos_j = item_pos_j

                if item_neg_i.shape[0] > max_negatives:
                    neg_indices = torch.randperm(item_neg_i.shape[0])[:max_negatives]
                    sampled_neg_j = item_neg_j[neg_indices]
                else:
                    sampled_neg_j = item_neg_j

                anchor_emb = item_emb_norm[anchor_pos_i]  # [min(N,1000), D]
                pos_emb = item_emb_norm[anchor_pos_j]     # [min(N,1000), D]
                neg_emb = item_emb_norm[sampled_neg_j]    # [min(M,1000), D]

                # Positive similarities
                pos_sim = torch.sum(anchor_emb * pos_emb, dim=-1) / tau  # [anchors]

                # Process in smaller batches to avoid OOM
                batch_size = 100
                loss_item = anchor_emb.new_tensor(0.0)
                for start_idx in range(0, anchor_emb.shape[0], batch_size):
                    end_idx = min(start_idx + batch_size, anchor_emb.shape[0])
                    batch_anchor = anchor_emb[start_idx:end_idx]  # [batch, D]
                    batch_pos_sim = pos_sim[start_idx:end_idx]    # [batch]

                    # Compute negative similarities for this batch
                    batch_neg_sim = torch.mm(batch_anchor, neg_emb.t()) / tau  # [batch, negatives]

                    # Vectorized InfoNCE for this batch
                    all_scores = torch.cat([batch_pos_sim.unsqueeze(1), batch_neg_sim], dim=1)
                    log_denominator = torch.logsumexp(all_scores, dim=1)
                    loss_item = loss_item + (-batch_pos_sim + log_denominator).sum()

                loss_item = loss_item / anchor_emb.shape[0]  # average over processed anchors

        loss = loss_bpr + args.item_pair_weight * loss_item
        return loss + args.lambda_reg * reg_loss, loss_bpr, loss_item
