"""Define biased matrix factorization and a small nonlinear rating model.

Both models learn user/movie embeddings. Matrix factorization uses a dot
product; the neural variant adds an MLP interaction term to that same baseline.
"""

import torch
from torch import nn


class RatingModel(nn.Module):
    """Predict explicit ratings from user and movie embeddings.

    Attributes:
        users: User embeddings of shape (number_of_users, dimensions).
        items: Movie embeddings of shape (number_of_movies, dimensions).
        user_bias: Learned per-user rating offsets.
        item_bias: Learned per-movie rating offsets.
        interaction: Optional MLP over user, movie, and elementwise product features.

    Scores are unbounded during training and clipped to [0.5, 5] for reporting.
    The neural residual makes the scorer nonlinear; the MF variant alone is
    a latent-factor model, not a deep network.
    """

    def __init__(self, users: int, items: int, mean: float, kind: str = 'mf', dimensions: int = 32):
        """Initialize embeddings, zero biases, global mean, and an optional neural head."""
        super().__init__()
        if kind not in ('mf', 'neural'):
            raise ValueError('kind must be mf or neural.')
        self.kind = kind
        self.users = nn.Embedding(users, dimensions)
        self.items = nn.Embedding(items, dimensions)
        self.user_bias = nn.Embedding(users, 1)
        self.item_bias = nn.Embedding(items, 1)
        self.register_buffer('mean', torch.tensor(float(mean)))
        nn.init.normal_(self.users.weight, std=0.05)
        nn.init.normal_(self.items.weight, std=0.05)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        self.interaction = (
            nn.Sequential(
                nn.Linear(dimensions * 3, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
            if kind == 'neural'
            else None
        )

    def score_vectors(self, user_vectors, user_bias, item_indices):
        """Score known items for arbitrary user vectors, including newly fitted viewers.

        Args:
            user_vectors: Embedding tensor shaped (N, D) or (1, D) for broadcasting.
            user_bias: User offsets broadcastable to a length-N vector.
            item_indices: Length-N tensor of movie row indices.

        Returns:
            Length-N raw predicted ratings. This path permits optimizing a new
            viewer embedding while leaving the trained model unchanged.
        """
        movie_vectors = self.items(item_indices)
        user_vectors = user_vectors.expand_as(movie_vectors)
        score = self.mean + user_bias.reshape(-1) + self.item_bias(item_indices).squeeze(-1)
        score = score + (user_vectors * movie_vectors).sum(dim=-1)
        if self.interaction is not None:
            combined = torch.cat(
                [user_vectors, movie_vectors, user_vectors * movie_vectors], dim=-1
            )
            score = score + self.interaction(combined).squeeze(-1)
        return score

    def forward(self, user_indices, item_indices):
        """Return one rating prediction for each aligned user/movie index pair."""
        return self.score_vectors(
            self.users(user_indices), self.user_bias(user_indices).squeeze(-1), item_indices
        )
