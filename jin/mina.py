import torch
import torch.nn.functional as F

class MINA:
    """
    PyTorch Implementation of the MINA Framework.
    """
    def __init__(self, embeddings: torch.Tensor, word_to_idx: dict, device: str = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.embeddings = embeddings.to(self.device)
        self.word_to_idx = word_to_idx
        self.pillar = None
        self.c_semantic = None

    def induce_pillar(self, stable_words: list):
        valid_words = [w for w in stable_words if w in self.word_to_idx]
        if not valid_words:
            raise ValueError("No valid words found in the embedding vocabulary.")
        indices = [self.word_to_idx[w] for w in valid_words]
        vectors = self.embeddings[indices]
        self.pillar = F.normalize(vectors.mean(dim=0), p=2, dim=0)
        return valid_words

    def decompose(self, word: str):
        if self.pillar is None:
            raise RuntimeError("Pillar not induced. Run induce_pillar first.")
        if word not in self.word_to_idx:
            return None, None
        vec = self.embeddings[self.word_to_idx[word]]
        h_val = torch.dot(vec, self.pillar)
        h_vec = h_val * self.pillar
        b_vec = vec - h_vec
        b_norm = torch.norm(b_vec, p=2)
        return h_val, b_norm

    def optimize_c(self, stable_words: list, lr: float = 0.01, epochs: int = 500):
        valid_words = [w for w in stable_words if w in self.word_to_idx]
        h_vals, b_norms = [], []
        for w in valid_words:
            h, b = self.decompose(w)
            h_vals.append(h)
            b_norms.append(b)
        h_vals = torch.stack(h_vals)
        b_norms = torch.stack(b_norms)

        c_init = (b_norms / h_vals).mean().item()
        c = torch.tensor([c_init], requires_grad=True, device=self.device)
        optimizer = torch.optim.Adam([c], lr=lr)

        for _ in range(epochs):
            optimizer.zero_grad()
            loss = torch.sum((b_norms - c * h_vals) ** 2)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                c.clamp_(min=1e-4)

        self.c_semantic = c.item()
        return self.c_semantic

    def calculate_mass(self, word: str):
        if self.c_semantic is None:
            raise RuntimeError("c_semantic not optimized.")
        h_val, b_norm = self.decompose(word)
        if h_val is None:
            return None
        mass_sq = (b_norm ** 2) - (self.c_semantic * h_val) ** 2
        m_val = torch.sign(mass_sq) * torch.sqrt(torch.abs(mass_sq))
        return {'Word': word, 'H': h_val.item(), 'B': b_norm.item(), 'M': m_val.item()}