import torch.nn as nn
import torch.nn.functional as F


class LearnableProjector(nn.Module):
    def __init__(self, input_dim=300, output_dim=3):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim, bias=False)
        nn.init.orthogonal_(self.proj.weight)

    def forward(self, x, condition=None):
        return self.proj(x)


class LearnableProjectorBias(nn.Module):
    def __init__(self, input_dim=300, output_dim=3):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim, bias=True)
        nn.init.orthogonal_(self.proj.weight)

    def forward(self, x, condition=None):
        return self.proj(x)


class LearnableProjectorGamma(nn.Module):
    def __init__(self, input_dim=300, condition_dim=3, output_dim=3):
        super().__init__()
        self.gamma = nn.Linear(condition_dim, input_dim)
        self.proj = nn.Linear(input_dim, output_dim, bias=False)
        nn.init.orthogonal_(self.proj.weight)

    def forward(self, x, condition=None):
        if condition is not None:
            gamma = self.gamma(condition)
            x = gamma * x

        return self.proj(x)


class LearnableProjectorGammaBeta(nn.Module):
    def __init__(self, input_dim=300, condition_dim=3, output_dim=3):
        super().__init__()
        self.gamma = nn.Linear(condition_dim, input_dim)
        self.beta = nn.Linear(condition_dim, input_dim)
        self.proj = nn.Linear(input_dim, output_dim, bias=False)
        nn.init.orthogonal_(self.proj.weight)

    def forward(self, x, condition=None):
        if condition is not None:
            gamma = self.gamma(condition)
            beta = self.beta(condition)
            x = (gamma * x) + beta
        return self.proj(x)


REGISTRY = {
    "linear": LearnableProjector,
    'linear_bias': LearnableProjectorBias,
    "film_gamma":   LearnableProjectorGamma,
    'film_gamma_beta': LearnableProjectorGammaBeta
}


def build_model(name: str, **kwargs) -> nn.Module:
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(REGISTRY)}")
    return REGISTRY[name](**kwargs)
