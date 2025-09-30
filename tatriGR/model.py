import torch
import torch.nn.functional as F
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType
from .encoders import TripartiteLightGraph


class TaTriGR(GeneralRecommender):
    """Tripartite LightGraph + BPR + IPD + LC."""
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset, n_cri: int):
        super().__init__(config, dataset)
        self.n_cri = n_cri
        self.n_subcri = n_cri - 1

        self.latent_dim = int(config.get('embedding_size', 64))
        self.n_layers = int(config.get('n_layers', 2))
        self.reg_weight = float(config.get('reg_weight', 1e-4))
        self.eval_chunk_size = int(config.get('eval_chunk_size', 4096))

        self.aux_ipd_weight = float(config.get('aux_ipd_weight', 0.0))
        self.aux_lexi_weight = float(config.get('aux_lexi_weight', 0.0))

        self.encoder = TripartiteLightGraph(
            n_users=self.n_users, n_items=self.n_items, n_subcri=self.n_subcri,
            embed_dim=self.latent_dim, n_layers=self.n_layers, l2norm=False
        )

        self.bpr_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        self.encoder.adj = None
        self.user_w_uc = None
        self.item_tilde_ic = None

        self._cache_user = None
        self._cache_item = None

    # Core score
    def _score(self, user, item):
        u_z, i_z, _ = self.encoder()
        return (u_z[user] * i_z[item]).sum(-1)

    # IPD: align e_u with ideal point sum_c w_uc e_c
    def _ipd_loss(self, user_idx, u_z, c_z):
        if self.aux_ipd_weight <= 0 or self.user_w_uc is None:
            return u_z.new_tensor(0.0)
        w = self.user_w_uc[user_idx]
        u_star = torch.matmul(w, c_z)
        u = u_z[user_idx]
        return F.mse_loss(u, u_star)

    # LC: use smoothed item-criterion score tilde r_ic* as delta
    def _lexi_loss(self, user_idx, pos_item_idx, neg_item_idx, u_z, i_z):
        if self.aux_lexi_weight <= 0 or self.user_w_uc is None or self.item_tilde_ic is None:
            return u_z.new_tensor(0.0)

        w = self.user_w_uc[user_idx]
        c_star = torch.argmax(w, dim=1)

        tilde_pos = self.item_tilde_ic[pos_item_idx, c_star]
        tilde_neg = self.item_tilde_ic[neg_item_idx, c_star]
        delta_r = tilde_pos - tilde_neg

        u = u_z[user_idx]
        i_pos = i_z[pos_item_idx]
        i_neg = i_z[neg_item_idx]
        y_pos = (u * i_pos).sum(-1)
        y_neg = (u * i_neg).sum(-1)

        return F.softplus(- delta_r * (y_pos - y_neg)).mean()

    def calculate_loss(self, interaction):
        self._cache_user, self._cache_item = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        u_z, i_z, c_z = self.encoder()

        # BPR
        y_pos = (u_z[user] * i_z[pos_item]).sum(-1)
        y_neg = (u_z[user] * i_z[neg_item]).sum(-1)
        loss_bpr = self.bpr_loss(y_pos, y_neg)

        # L2 regularization
        emb_reg = self.reg_loss(
            self.encoder.user(user),
            self.encoder.item(pos_item),
            self.encoder.item(neg_item),
            self.encoder.crit.weight
        )
        loss_reg = self.reg_weight * emb_reg

        # Aux losses
        loss_ipd = self._ipd_loss(user, u_z, c_z) * self.aux_ipd_weight
        loss_lexi = self._lexi_loss(user, pos_item, neg_item, u_z, i_z) * self.aux_lexi_weight

        return loss_bpr + loss_reg + loss_ipd + loss_lexi

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        return self._score(user, item)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self._cache_user is None or self._cache_item is None:
            u_z, i_z, _ = self.encoder()
            self._cache_user, self._cache_item = u_z, i_z
        u = self._cache_user[user]
        scores = torch.matmul(u, self._cache_item.transpose(0, 1)).view(-1)
        return scores
