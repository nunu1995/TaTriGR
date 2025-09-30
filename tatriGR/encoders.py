import torch
from torch import nn
import torch.nn.functional as F


class TripartiteLightGraph(nn.Module):
    """Lightweight propagation over the tripartite graph with ELU activation."""
    def __init__(self, n_users, n_items, n_subcri, embed_dim=64, n_layers=2, l2norm=False):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_subcri = n_subcri
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.l2norm = l2norm

        self.user = nn.Embedding(n_users, embed_dim)
        self.item = nn.Embedding(n_items, embed_dim)
        self.crit = nn.Embedding(n_subcri, embed_dim)

        nn.init.xavier_uniform_(self.user.weight)
        nn.init.xavier_uniform_(self.item.weight)
        nn.init.xavier_uniform_(self.crit.weight)

        self.activation = nn.ELU()
        self.adj = None

    def propagate(self, all_embed):
        out = [all_embed]
        x = all_embed
        for _ in range(self.n_layers):
            x = torch.sparse.mm(self.adj, x)
            x = self.activation(x)
            out.append(x)
        x = torch.stack(out, dim=1).mean(1)
        if self.l2norm:
            x = F.normalize(x, dim=-1)
        return x

    def forward(self):
        assert self.adj is not None, "TripartiteLightGraph.adj is not set."
        ui = torch.cat([self.user.weight, self.item.weight], dim=0)
        all0 = torch.cat([ui, self.crit.weight], dim=0)
        z = self.propagate(all0)
        u_z = z[:self.n_users]
        i_z = z[self.n_users:self.n_users + self.n_items]
        c_z = z[self.n_users + self.n_items:]
        return u_z, i_z, c_z
