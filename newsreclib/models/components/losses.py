import torch
from pytorch_metric_learning import losses
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu


class SupConLoss(losses.SupConLoss):
    def __init__(self, temperature=0.1, **kwargs):
        super().__init__()
        self.temperature = temperature
        self.add_to_recordable_attributes(list_of_names=["temperature"], is_stat=False)

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        # Overwrite original method to use directly similarity matrix instead of embeddings
        if all(len(x) <= 1 for x in indices_tuple):
            return self.zero_losses()
        mat = embeddings
        return self.loss_method(mat, indices_tuple)

    def _compute_loss(self, mat, pos_mask, neg_mask):
        if pos_mask.bool().any() and neg_mask.bool().any():
            mat = mat / self.temperature
            mat_max, _ = mat.max(dim=1, keepdim=True)
            mat = mat - mat_max.detach()  # for numerical stability

            denominator = lmu.logsumexp(
                mat, keep_mask=(pos_mask + neg_mask).bool(), add_one=False, dim=1
            )
            log_prob = mat - denominator
            mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (
                pos_mask.sum(dim=1) + c_f.small_val(mat.dtype)
            )

            return {
                "loss": {
                    "losses": -mean_log_prob_pos,
                    "indices": c_f.torch_arange_from_size(mat),
                    "reduction_type": "element",
                }
            }
        return self.zero_losses()


class BPRLoss(losses.BaseMetricLossFunction):
    """Bayesian Personalized Ranking Loss

    References:
    - https://d2l.ai/chapter_recommender-systems/ranking.html
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        # Unpack indices_tuple which is expected to contain (q_p, p, q_n, n)
        scores = embeddings
        q_p, p, q_n, n = indices_tuple

        # Gathering positive scores
        pos_scores = scores[q_p, p]

        #  Initialize an empty tensor for collecting individual user losses
        user_losses = torch.tensor([], device=scores.device, dtype=scores.dtype)

        # Loop over each positive score index
        for idx, pos_idx in enumerate(q_p):
            # Find indices where the negative scores match the current positive score's user index
            relevant_neg_idx = q_n == pos_idx

            # Gather negative scores for the current user using the filtered indices
            neg_scores_for_user = scores[pos_idx, n[relevant_neg_idx]]

            # Calculate differences between the current positive score and all corresponding negatives
            differences = pos_scores[idx] - neg_scores_for_user

            # Apply sigmoid to the differences
            sigmoid_differences = torch.sigmoid(differences)

            # Calculate the negative log of the sigmoid of differences
            losses = -torch.log(
                sigmoid_differences + torch.finfo(scores.dtype).eps
            )  # Adding a small epsilon for numerical stability

            # Concatenate the current user's losses to the total
            user_losses = torch.cat((user_losses, losses))

        # Calculate the average loss
        mean_loss = torch.mean(losses)

        return {
            "loss": {
                "losses": mean_loss,
                "indices": None,
                "reduction_type": "already_reduced",
            }
        }
