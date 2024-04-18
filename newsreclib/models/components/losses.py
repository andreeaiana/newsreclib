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
    """ Bayesian Personalized Ranking Loss

    References:
    - https://d2l.ai/chapter_recommender-systems/ranking.html
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_loss(self, scores, labels, indices_tuple, ref_emb, ref_labels):
        # Unpack indices_tuple which is expected to contain (q_p, p, q_n, n)
        q_p, p, q_n, n = indices_tuple

        print("------ embeddings ------")
        print(scores.size())
        print("------ q_p ------")
        print(q_p.size())
        print("------ p ------")
        print(p.size())
        print("------ q_n ------")
        print(q_n.size())
        print("------ n ------")
        print(n.size())
        print("-----------")

        # Extract scores for positive and negative samples using provided indices
        pos_scores = scores[q_p, p]
        neg_scores = scores[q_n, n]

        print("------ pos_scores ------")
        print(pos_scores.size())
        print(pos_scores.unsqueeze(1).size())
        print("------ neg_scores ------")
        print(neg_scores.size())
        print(neg_scores.unsqueeze(0).size())
        print("-----------")

        # Compute the score differences
        score_diffs = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)

        # Apply the sigmoid function to the score differences
        sigmoid_scores = torch.sigmoid(score_diffs)

        # Compute the log likelihood of the sigmoid scores
        losses = -torch.log(sigmoid_scores + c_f.small_val(scores.dtype))  # Add epsilon for numerical stability

        # Calculate the average loss
        mean_loss = torch.mean(losses)

        return {
            "loss": {
                "losses": mean_loss,
                "indices": None,
                "reduction_type": "already_reduced",
            }
        }
