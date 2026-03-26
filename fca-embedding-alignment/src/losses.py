import torch
import torch.nn.functional as F


def alignment_loss(projector, vec_a, vec_b, target_axis):
    z_a = projector(vec_a, target_axis)
    z_b = projector(vec_b, target_axis)
    cos_sim = F.cosine_similarity(z_a - z_b, target_axis, dim=1)
    return (1 - cos_sim).mean()


def orthogonality_loss(projector):
    W = projector.proj.weight
    W_norm = F.normalize(W, dim=1)
    cos_matrix = W_norm @ W_norm.T
    mask = ~torch.eye(W_norm.shape[0], dtype=torch.bool, device=W.device)
    return (cos_matrix[mask] ** 2).mean()


def total_loss(projector, vec_a, vec_b, target_axis, lambda_ortho=0.1):
    l_align = alignment_loss(projector, vec_a, vec_b, target_axis)
    l_ortho = orthogonality_loss(projector)
    return l_align + lambda_ortho * l_ortho, l_align, l_ortho
