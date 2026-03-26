import numpy as np
import torch
from tqdm.auto import tqdm
from .losses import alignment_loss, orthogonality_loss, total_loss


def train(projector, train_loader, val_loader, optimizer,
          lambda_ortho=0.1, num_epochs=1500, device="cpu"):

    projector.to(device)
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    history = {
        "train_loss": [], "val_loss": [],
        "train_align": [], "val_align": [],
        "ortho": [],
    }

    for epoch in tqdm(range(1, num_epochs + 1), desc="Training"):
        projector.train()
        t_losses, t_aligns = [], []
        for vec_a, vec_b, target in train_loader:
            vec_a, vec_b, target = vec_a.to(
                device), vec_b.to(device), target.to(device)
            l_align = alignment_loss(projector, vec_a, vec_b, target)
            l_ortho = orthogonality_loss(projector)
            loss = l_align + lambda_ortho * l_ortho
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            t_losses.append(loss.item())
            t_aligns.append(l_align.item())

        projector.eval()
        v_losses, v_aligns = [], []
        with torch.no_grad():
            for vec_a, vec_b, target in val_loader:
                vec_a, vec_b, target = vec_a.to(
                    device), vec_b.to(device), target.to(device)
                l_align = alignment_loss(projector, vec_a, vec_b, target)
                l_ortho = orthogonality_loss(projector)
                v_losses.append((l_align + lambda_ortho * l_ortho).item())
                v_aligns.append(l_align.item())

        v_loss = np.mean(v_losses)
        history["train_loss"].append(np.mean(t_losses))
        history["val_loss"].append(v_loss)
        history["train_align"].append(np.mean(t_aligns))
        history["val_align"].append(np.mean(v_aligns))
        history["ortho"].append(l_ortho.item())

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_state = {k: v.clone()
                          for k, v in projector.state_dict().items()}
            best_epoch = epoch

    projector.load_state_dict(best_state)
    print(f"\nBest model: epoch {best_epoch}, val_loss={best_val_loss:.4f}")
    return projector, history

# thought about triplet loss (https://en.wikipedia.org/wiki/Triplet_loss#References) maybe we can add it as an additional loss term to further encourage the model to learn better representations. The triplet loss would require us to have triplets of data (anchor, positive, negative) during training, which might be a bit more complex to set up but could potentially improve the performance of the projector.
# like total = alignment_loss + lambda_ortho * orthogonality_loss + lambda_triplet * triplet_loss
