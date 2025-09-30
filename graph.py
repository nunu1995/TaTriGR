import numpy as np
import torch
import scipy.sparse as sp
import pandas as pd
from typing import Tuple


def build_tripartite_adj_and_weights(
    ui_pairs_internal,
    df_train_plain: pd.DataFrame,
    n_users: int, n_items: int, criteria: list,
    uc_alpha: float = 2.0, ic_beta: float = 10.0,
    lambda_ui: float = 1.0, lambda_uc: float = 1.0, lambda_ic: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct tripartite adjacency with weights.
    Returns:
      - adj: normalized sparse adjacency
      - w_uc: [U,C] user–criterion weights
      - tilde_ic: [I,C] smoothed item–criterion scores
    """
    sub_criteria = criteria[:-1]
    overall_col = criteria[-1]
    C = len(sub_criteria)
    U, I = n_users, n_items
    N = U + I + C

    rows, cols, vals = [], [], []

    # U–I: min–max normalization of overall
    ui = df_train_plain[['user_id', 'item_id', overall_col]].copy()
    rmin, rmax = ui[overall_col].min(), ui[overall_col].max()
    ui_w = {}
    for _, row in ui.iterrows():
        u, it, r = int(row['user_id']), int(row['item_id']), float(row[overall_col])
        ui_w[(u, it)] = (r - rmin) / (rmax - rmin) if rmax > rmin else 1.0

    for (u, it) in ui_pairs_internal:
        w = float(ui_w.get((u, it), 1.0)) * float(lambda_ui)
        rows.append(u);     cols.append(U + it); vals.append(w)
        rows.append(U + it); cols.append(u);     vals.append(w)

    # U–C: centered & variance-shrunk + softmax(alpha)
    ug_mean = (df_train_plain
               .groupby('user_id')[sub_criteria]
               .mean()
               .reindex(range(U))
               .fillna(0.0)
               .to_numpy(np.float32))
    var = (df_train_plain
           .groupby('user_id')[sub_criteria]
           .var()
           .reindex(range(U))
           .fillna(0.0)
           .to_numpy(np.float32))
    s_uc = np.sqrt(np.clip(var, 0.0, None))
    rbar_u = ug_mean.mean(axis=1, keepdims=True)
    p_uc = (ug_mean - rbar_u) * (1.0 / (1.0 + s_uc))
    x = uc_alpha * p_uc
    x = x - x.max(axis=1, keepdims=True)
    ex = np.exp(x)
    w_uc = ex / (ex.sum(axis=1, keepdims=True) + 1e-8)
    if lambda_uc != 1.0:
        w_uc = w_uc * float(lambda_uc)

    baseC = U + I
    for u_id in range(U):
        for c_idx in range(C):
            w = float(w_uc[u_id, c_idx])
            if w <= 0: continue
            c_node = baseC + c_idx
            rows.append(u_id)
            cols.append(c_node)
            vals.append(w)
            rows.append(c_node)
            cols.append(u_id)
            vals.append(w)

    # I–C: Bayesian smoothing + min–max
    ig_mean = (df_train_plain
               .groupby('item_id')[sub_criteria]
               .mean()
               .reindex(range(I))
               .fillna(0.0)
               .to_numpy(np.float32))
    ig_cnt  = (df_train_plain
               .groupby('item_id')[sub_criteria]
               .count()
               .reindex(range(I))
               .fillna(0)
               .to_numpy(np.float32))
    rc_mean = df_train_plain[sub_criteria].mean(axis=0).to_numpy(np.float32)

    tilde_ic = (ig_cnt * ig_mean + ic_beta * rc_mean) / (ig_cnt + ic_beta + 1e-8)
    mn = tilde_ic.min(axis=0, keepdims=True)
    mx = tilde_ic.max(axis=0, keepdims=True)
    w_ic = (tilde_ic - mn) / (mx - mn + 1e-8)
    if lambda_ic != 1.0:
        w_ic = w_ic * float(lambda_ic)

    for it in range(I):
        for c_idx in range(C):
            w = float(w_ic[it, c_idx])
            if w <= 0: continue
            c_node = baseC + c_idx
            rows.append(U + it)
            cols.append(c_node)
            vals.append(w)
            rows.append(c_node)
            cols.append(U + it)
            vals.append(w)

    # Sparse adjacency and symmetric normalization
    idx = np.vstack([rows, cols])
    A = sp.coo_matrix((vals, (idx[0], idx[1])), shape=(N, N), dtype=np.float32)
    deg = np.array(A.sum(axis=1)).ravel() + 1e-7
    D = sp.diags(np.power(deg, -0.5))
    L = D @ A @ D
    L = sp.coo_matrix(L)
    ij = torch.LongTensor([L.row, L.col])
    vv = torch.FloatTensor(L.data)
    adj = torch.sparse_coo_tensor(ij, vv, (N, N)).coalesce()

    return adj, torch.from_numpy(w_uc), torch.from_numpy(tilde_ic)
