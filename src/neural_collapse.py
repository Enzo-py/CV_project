import torch
import torch.nn.functional as F

def compute_nc_metrics(features, labels, weights=None, num_classes=100):
    """
    Calcule les indicateurs de Neural Collapse.
    features: [N, d], labels: [N], weights: [K, d] (Optionnel)
    """
    K = num_classes
    d = features.shape[1]
    
    class_indices = [(labels == i).nonzero(as_tuple=True)[0] for i in range(K)]
    means = torch.stack([features[idx].mean(dim=0) for idx in class_indices]) # mu_c
    global_mean = features.mean(dim=0) # mu_g
    centered_means = means - global_mean
    
    # --- NC1: Variability Collapse ---
    sw = torch.tensor(0.0)
    for i, idx in enumerate(class_indices):
        diff = features[idx] - means[i]
        sw += torch.sum(torch.norm(diff, dim=1)**2)
    sw /= features.shape[0]
    sb = torch.sum(torch.norm(centered_means, dim=1)**2) / K
    nc1 = sw / sb

    # --- NC2: Simplex Structure ---
    norm_means = F.normalize(centered_means, p=2, dim=1)
    cos_sim = torch.mm(norm_means, norm_means.T)
    off_diag = cos_sim[~torch.eye(K, dtype=bool)]
    nc2 = torch.std(off_diag - (-1.0 / (K - 1)))

    # --- NC3: Self Alignment ---
    nc3 = torch.tensor(0.0)
    if weights is not None:
        w_norm = F.normalize(weights, p=2, dim=1)
        m_norm = F.normalize(centered_means, p=2, dim=1)
        nc3 = torch.norm(w_norm - m_norm)**2 / K

    # --- NC4: Decision Equivalence ---
    dist = torch.cdist(features, means)
    preds_ncc = dist.argmin(dim=1)
    nc4 = (preds_ncc == labels).float().mean()

    # --- NC5: Norm Convergence ---
    nc5 = torch.tensor(0.0)
    if weights is not None:
        w_norms = torch.norm(weights, dim=1)
        nc5 = w_norms.std() / w_norms.mean()

    return {"NC1": nc1.item(), "NC2": nc2.item(), "NC3": nc3.item(), "NC4": nc4.item(), "NC5": nc5.item()}