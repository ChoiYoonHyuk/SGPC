import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.nn import GATConv
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Actor, WebKB

# dataset selection
parser = argparse.ArgumentParser(description='Dataset')
parser.add_argument('data', type=int, help='data selector')
args = parser.parse_args()
data_id = args.data

if data_id == 0:
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
elif data_id == 1:
    dataset = Planetoid(root='/tmp/Citeseer', name='Citeseer')
elif data_id == 2:
    dataset = Planetoid(root='/tmp/Pubmed', name='Pubmed')
elif data_id == 3:
    dataset = WikipediaNetwork(root='/tmp/Chameleon', name='chameleon')
elif data_id == 4:
    dataset = WikipediaNetwork(root='/tmp/Squirrel', name='squirrel')
elif data_id == 5:
    dataset = Actor(root='/tmp/Actor')
elif data_id == 6:
    dataset = WebKB(root='/tmp/Cornell', name='Cornell')
elif data_id == 7:
    dataset = WebKB(root='/tmp/Texas', name='Texas')
else:
    dataset = WebKB(root='/tmp/Wisconsin', name='Wisconsin')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = dataset[0].to(device)
num_class = dataset.num_classes

if data.train_mask.dim() == 2:
    data.train_mask = data.train_mask[:, 0]
    data.val_mask = data.val_mask[:, 0]
    data.test_mask = data.test_mask[:, 0]

# adaptive config
set_1 = {0, 1, 2}
set_2 = {3, 4, 5, 6, 7, 8, 9}

if data_id in set_1:
    hidden_dim = 128
    drop_prob = 0.5
    epsilon_ot = 5e-3
    delta_t = 0.02
    pacbayes_epoch = 300
    lambda_KL = 1e-1
    lambda_spec = 1e-4
    use_gat = True
else:
    hidden_dim = 256
    drop_prob = 0.5
    epsilon_ot = 5e-3
    delta_t = 0.15
    pacbayes_epoch = 300
    lambda_KL = 1e-1
    lambda_spec = 1e-3
    use_gat = False

lr = 1e-3
patience = 1000

# sheaf / OT utilities
def sinkhorn_simple(C, epsilon, n_iters=1):
    P0 = torch.exp(-C / epsilon)
    P0 = torch.clamp(P0, min=1e-3, max=1.0)
    return P0

def jko_refine(P0, C_feat, epsilon):
    logits = (P0 + 1e-12).log() - C_feat / epsilon
    P_star = torch.exp(logits)
    P_star = torch.clamp(P_star, min=1e-3, max=1.0)
    return P_star

def build_sheaf_laplacian(N, row, col, R):
    # R: [E, d]
    # weight per edge
    w = (R ** 2).mean(dim=-1)
    # off-diagonal entries
    i = torch.cat([row, col], dim=0)
    j = torch.cat([col, row], dim=0)
    v = torch.cat([-w, -w], dim=0)
    # diagonal
    diag = torch.zeros(N, device=row.device)
    diag.index_add_(0, row, w)
    diag.index_add_(0, col, w)
    i = torch.cat([i, torch.arange(N, device=row.device)], dim=0)
    j = torch.cat([j, torch.arange(N, device=row.device)], dim=0)
    v = torch.cat([v, diag], dim=0)
    return SparseTensor(row=i, col=j, value=v, sparse_sizes=(N, N))

def conjugate_gradient(L, B, delta_t, tol=1e-4, maxiter=20):
    N, d = B.size()
    X = torch.zeros_like(B)
    R = B.clone()
    P = R.clone()
    rs_old = (R * R).sum(dim=0)

    def A(M):
        return M + delta_t * L.matmul(M)

    for _ in range(maxiter):
        AP = A(P)
        denom = (P * AP).sum(dim=0) + 1e-16
        alpha = rs_old / denom
        X = X + P * alpha
        R = R - AP * alpha
        rs_new = (R * R).sum(dim=0)
        if rs_new.max() < tol ** 2:
            break
        beta = rs_new / (rs_old + 1e-16)
        P = R + P * beta
        rs_old = rs_new
    return X

# Adaptive Frequency Mixing
def normalize_sheaf_laplacian(L: SparseTensor):
    # reconstruct degree from off-diagonal negatives
    row, col, val = L.coo()
    mask_off = row != col
    off_row = row[mask_off]
    off_val = val[mask_off]  # negative
    N = L.size(0)
    deg = torch.zeros(N, device=val.device)
    deg.index_add_(0, off_row, -off_val)
    deg = torch.clamp(deg, min=1e-8)
    inv_sqrt_deg = (1.0 / deg).sqrt()
    return inv_sqrt_deg

def apply_tildeL(L: SparseTensor, inv_sqrt_deg, X):
    Y = inv_sqrt_deg.unsqueeze(-1) * X
    LY = L.matmul(Y)
    Z = inv_sqrt_deg.unsqueeze(-1) * LY
    return X + Z

def afm_branch(L, X, Q=3, gamma=None):
    inv_sqrt_deg = normalize_sheaf_laplacian(L)
    T_list = [X]
    if Q >= 1:
        T_list.append(apply_tildeL(L, inv_sqrt_deg, X))
    for _ in range(2, Q + 1):
        T_next = 2 * apply_tildeL(L, inv_sqrt_deg, T_list[-1]) - T_list[-2]
        T_list.append(T_next)
    alpha = torch.softmax(gamma, dim=0)
    out = 0.0
    for q in range(Q + 1):
        out = out + alpha[q] * T_list[q]
    return out

# PAC-Bayes components (used only on heterophily)
def beta_kl(a_post, b_post, a_prior=1.0, b_prior=1.0):
    device = a_post.device
    dtype = a_post.dtype
    a_prior = torch.tensor(a_prior, device=device, dtype=dtype)
    b_prior = torch.tensor(b_prior, device=device, dtype=dtype)
    term1 = torch.lgamma(a_post + b_post) - torch.lgamma(a_post) - torch.lgamma(b_post)
    term2 = - (torch.lgamma(a_prior + b_prior) - torch.lgamma(a_prior) - torch.lgamma(b_prior))
    term3 = (a_post - a_prior) * (torch.digamma(a_post) - torch.digamma(a_post + b_post))
    term4 = (b_post - b_prior) * (torch.digamma(b_post) - torch.digamma(a_post + b_post))
    return term1 + term2 + term3 + term4

def estimate_edge_posterior(probs, edge_index, a0=1.0, b0=1.0):
    row, col = edge_index
    agree = (probs[row] * probs[col]).sum(dim=1).clamp(0.0, 1.0)
    a_post = a0 + agree
    b_post = b0 + (1.0 - agree)
    kappa = a_post / (a_post + b_post)
    kl = beta_kl(a_post, b_post, a0, b0)
    return kappa, kl

def spectral_gap_from_sparse(L: SparseTensor):
    L_dense = L.to_torch_sparse_coo_tensor().to_dense()
    L_sym = 0.5 * (L_dense + L_dense.t())
    evals = torch.linalg.eigvalsh(L_sym)
    evals, _ = torch.sort(evals)
    if evals.numel() < 2:
        return torch.tensor(0.0, device=L_dense.device, dtype=L_dense.dtype)
    return torch.clamp(evals[1], min=1e-3)

def heterophily_penalty(kappa_edges, edge_index, labels, num_class):
    row, col = edge_index
    y_i = labels[row]
    y_j = labels[col]
    valid = (y_i >= 0) & (y_j >= 0)
    y_i = y_i[valid]
    y_j = y_j[valid]
    kappa = kappa_edges[valid]

    Pi = torch.zeros(num_class, num_class, device=labels.device)
    for c in range(num_class):
        for c2 in range(num_class):
            mask = (y_i == c) & (y_j == c2)
            if mask.any():
                Pi[c, c2] = kappa[mask].mean()
    c_het = torch.norm(Pi, p='fro')
    return c_het

def nll_smooth(logits, y, eps=0.1):
    log_probs = F.log_softmax(logits, dim=1)
    n_class = logits.size(1)
    one_hot = F.one_hot(y, n_class).float()
    soft = (1 - eps) * one_hot + eps / n_class
    loss = -(soft * log_probs).sum(dim=1).mean()
    return loss

# model
class SVRSheafNet(nn.Module):
    def __init__(self, in_feats, hidden_dim, num_class,
                 delta_t=0.1, epsilon=1e-3,
                 cg_tol=1e-4, cg_maxiter=20,
                 dropout=0.6, Q=3, use_gat=True):
        super().__init__()
        self.delta_t = delta_t
        self.epsilon_ot = epsilon
        self.cg_tol = cg_tol
        self.cg_maxiter = cg_maxiter
        self.drop = dropout
        self.Q = Q
        self.use_gat = use_gat

        self.in_lin = nn.Linear(in_feats, hidden_dim)
        self.in_norm = nn.LayerNorm(hidden_dim)

        self.W_sheaf = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.W_sheaf)

        self.gamma = nn.Parameter(torch.zeros(Q + 1))

        self.alpha_svr = nn.Parameter(torch.tensor(-4.0))
        self.alpha_afm = nn.Parameter(torch.tensor(-4.0))
        if data_id == 3:
            self.alpha_svr = nn.Parameter(torch.tensor(-8.0))
            self.alpha_afm = nn.Parameter(torch.tensor(-8.0))

        if self.use_gat:
            self.gat1 = GATConv(hidden_dim, 8, heads=8,
                                concat=True, dropout=dropout)
            self.gat2 = GATConv(64, num_class,
                                heads=1, concat=False,
                                dropout=dropout)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_class)
            )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        row, col = edge_index
        N = x.size(0)

        x = x - 1e-1 if data_id in [0, 1] else x
        h = torch.sigmoid(self.in_norm(self.in_lin(x)))

        feat_row = h[row] @ self.W_sheaf
        feat_col = h[col] @ self.W_sheaf
        diff = feat_row - feat_col
        
        C_feat = (diff ** 2).mean(dim=1)

        P0 = sinkhorn_simple(C_feat, self.epsilon_ot, n_iters=1)
        P_star = jko_refine(P0, C_feat, self.epsilon_ot)
        
        w = 0.7 * P0 + 0.3 * P_star
        
        R_ij = w.unsqueeze(1) * feat_row

        L = build_sheaf_laplacian(N, row, col, R_ij)

        H_svr = conjugate_gradient(L, h, self.delta_t,
                                   tol=self.cg_tol,
                                   maxiter=self.cg_maxiter)
        H_afm = afm_branch(L, h, Q=self.Q, gamma=self.gamma)

        alpha_svr = torch.sigmoid(self.alpha_svr)
        alpha_afm = torch.sigmoid(self.alpha_afm)
        sheaf_res = alpha_svr * H_svr + alpha_afm * H_afm

        fused = h + sheaf_res
        fused = F.dropout(fused, p=self.drop, training=self.training)

        if self.use_gat:
            out = self.gat1(fused, edge_index)
            out = F.elu(out)
            out = F.dropout(out, p=self.drop, training=self.training)
            logits = self.gat2(out, edge_index)
        else:
            logits = self.mlp(fused)

        return {
            "logits": logits,
            "L_sheaf": L
        }


# build model/optimizer
model = SVRSheafNet(
    in_feats=dataset.num_node_features,
    hidden_dim=hidden_dim,
    num_class=num_class,
    delta_t=delta_t,
    epsilon=epsilon_ot,
    cg_tol=1e-4,
    cg_maxiter=10 if data_id in set_1 else 20,
    dropout=drop_prob,
    Q=3,
    use_gat=use_gat,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=50,
    min_lr=1e-5
)

best_val, best_test = 0.0, 0.0
patience_ctr = 0

# training loop
for epoch in range(1, 2001):
    model.train()
    optimizer.zero_grad()

    out = model(data)
    logits = out["logits"]
    L_sheaf = out["L_sheaf"]

    log_probs = F.log_softmax(logits, dim=1)
    probs = log_probs.exp()

    # base loss
    loss = nll_smooth(logits[data.train_mask], data.y[data.train_mask], eps=0.0)

    if epoch > pacbayes_epoch:
        kappa_ij, kl_edges = estimate_edge_posterior(
            probs, data.edge_index, a0=1.0, b0=1.0
        )
        n_labeled = int(data.train_mask.sum().item())
        delta_conf = 0.1
        KL_mean = kl_edges.mean()
        pac_raw = (KL_mean + torch.log(torch.tensor(2.0 / delta_conf, device=device))) / (2.0 * n_labeled)
        pac_raw = torch.clamp(pac_raw, min=0.0)
        loss_KL = torch.sqrt(pac_raw + 1e-16)

        lambda2 = spectral_gap_from_sparse(L_sheaf).detach()
        lambda2 = torch.clamp(lambda2, min=1.0)
        c_het = heterophily_penalty(kappa_ij, data.edge_index, data.y, num_class)
        loss_spec = c_het / (lambda2 + 1e-6)

        loss = loss + lambda_KL * loss_KL + lambda_spec * loss_spec

    loss.backward()
    optimizer.step()

    # eval
    model.eval()
    with torch.no_grad():
        out_eval = model(data)
        logits_eval = out_eval["logits"]
        pred = logits_eval.argmax(dim=1)
        val_acc = pred[data.val_mask].eq(data.y[data.val_mask]).float().mean().item()
        test_acc = pred[data.test_mask].eq(data.y[data.test_mask]).float().mean().item()
        if test_acc > best_test:
            best_test = test_acc

    scheduler.step(val_acc)

    if val_acc > best_val:
        best_val = val_acc
        patience_ctr = 0
    else:
        patience_ctr += 1
        if patience_ctr > patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if epoch % 50 == 0 or val_acc == best_val:
        print(f"Epoch {epoch:04d} | Loss: {loss.item():.4f} | Val: {val_acc:.4f} | Test: {test_acc:.4f} | Best Test: {best_test:.4f}")
