import numpy as np
import torch


ANALOGIES = [
    ("king",        "man",    "woman",    "queen"),
    ("prince",      "boy",    "girl",     "princess"),
    ("husband",     "man",    "woman",    "wife"),
    ("uncle",       "man",    "woman",    "aunt"),
    ("father",      "man",    "woman",    "mother"),
    ("brother",     "man",    "woman",    "sister"),
    ("king",        "prince", "princess", "queen"),
    ("grandfather", "father", "mother",   "grandmother"),
    ("actor",       "man",    "woman",    "actress"),
    ("he",          "man",    "woman",    "she"),
]


def project_word(word, embeddings, projector, condition=None, device="cpu"):
    vec = torch.tensor(
        embeddings[word], dtype=torch.float32).unsqueeze(0).to(device)
    cond = None
    if condition is not None:
        cond = torch.tensor(
            condition, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        return projector(vec, cond).squeeze(0).cpu().numpy()


def test_analogy(a, b, c, expected, embeddings, projector, device="cpu"):
    pa = project_word(a, embeddings, projector, device=device)
    pb = project_word(b, embeddings, projector, device=device)
    pc = project_word(c, embeddings, projector, device=device)
    pe = project_word(expected, embeddings, projector, device=device)
    predicted = pa - pb + pc
    dist = np.linalg.norm(predicted - pe)  # Euclidean distance
    cos = np.dot(predicted, pe) / (np.linalg.norm(predicted)
                                   # Cosine similarity
                                   * np.linalg.norm(pe) + 1e-10)
    print(f"  {a} - {b} + {c} → {expected}  |  cos={cos:.4f}  dist={dist:.4f}")
    return {"dist": dist, "cos": cos}


def axis_purity(pairs, embeddings, projector, target_axis, device="cpu"):
    scores = []
    target = np.array(target_axis, dtype=np.float32)
    target = target / (np.linalg.norm(target) + 1e-10)
    for w1, w2 in pairs:
        p1 = project_word(w1, embeddings, projector, device=device)
        p2 = project_word(w2, embeddings, projector, device=device)
        diff = p1 - p2
        norm = np.linalg.norm(diff)
        if norm < 1e-10:
            continue
        cos = np.dot(diff / norm, target)
        scores.append(cos)
    return float(np.mean(scores))


def cross_leakage(pairs, embeddings, projector, leakage_axis, device="cpu"):
    leak = np.array(leakage_axis, dtype=np.float32)
    leak = leak / (np.linalg.norm(leak) + 1e-10)
    scores = []
    for w1, w2 in pairs:
        p1 = project_word(w1, embeddings, projector, device=device)
        p2 = project_word(w2, embeddings, projector, device=device)
        diff = p1 - p2
        norm = np.linalg.norm(diff)
        if norm < 1e-10:
            continue
        scores.append(abs(np.dot(diff / norm, leak)))
    return float(np.mean(scores))


def run_all_evaluations(projector, embeddings, gender_pairs=None, royalty_pairs=None, device="cpu"):
    """
    METRICS:
      analogy_cos   
      gender_purity 
      status_purity 
      gender_leakage 
      status_leakage
    """
    print("Analigies:")
    results = []
    for a, b, c, expected in ANALOGIES:
        r = test_analogy(a, b, c, expected, embeddings, projector, device)
        results.append(r)
    mean_cos = float(np.mean([r["cos"] for r in results]))
    mean_dist = float(np.mean([r["dist"] for r in results]))
    print(f"  analogy_cos={mean_cos:.4f} | analogy_dist={mean_dist:.4f}")

    metrics = {"analogy_cos": mean_cos, "analogy_dist": mean_dist}

    if gender_pairs and royalty_pairs:
        print("\nQuality of Axes")
        g_purity = axis_purity(gender_pairs, embeddings,
                               projector, [1, 0, 0], device)
        s_purity = axis_purity(royalty_pairs, embeddings,
                               projector, [0, 1, 0], device)
        g_leak = cross_leakage(gender_pairs, embeddings,
                               projector, [0, 1, 0], device)
        s_leak = cross_leakage(royalty_pairs, embeddings,
                               projector, [1, 0, 0], device)
        print(f"  gender_purity={g_purity:.4f} | status_purity={s_purity:.4f}")
        print(f"  gender_leakage={g_leak:.4f} | status_leakage={s_leak:.4f}")
        metrics.update({
            "gender_purity": g_purity,
            "status_purity": s_purity,
            "gender_leakage": g_leak,
            "status_leakage": s_leak,
        })

    return metrics
