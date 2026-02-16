import torch

def score_msp(logits):
    """Max Softmax Probability"""
    probs = torch.softmax(logits, dim=1)
    return torch.max(probs, dim=1)[0]

def score_max_logit(logits):
    """Maximum Logit Score"""
    return torch.max(logits, dim=1)[0]

def score_energy(logits, temperature=1.0):
    """Energy Score: -T * log(sum(exp(logits/T)))"""
    return temperature * torch.logsumexp(logits / temperature, dim=1)

def score_mahalanobis(features, mean, precision):
    """
    Mahalanobis Distance: -(f - mu)^T * Sigma^-1 * (f - mu)
    Note: On prend le négatif car un score plus élevé = plus ID.
    """
    diff = features - mean
    term = torch.mm(diff, precision)
    score = -torch.sum(term * diff, dim=1)
    return score

def score_vim(features, logits, w, b, u, sigma_inv):
    """
    ViM (Visual Interaction Matrix): Combine Logit et distance résiduelle PCA.
    """
    features_centered = features - u
    v_proj = features_centered - (features_centered @ w.T @ w)
    residual_norm = torch.norm(v_proj, dim=1)
    
    energy = torch.logsumexp(logits, dim=1)
    
    return energy - residual_norm

class OODEvaluator:
    def __init__(self, model):
        self.model = model.eval()
        self.feature_dim = model.base_model.fc.in_features

    def get_features_and_logits(self, dataloader, device):
        all_features = []
        all_logits = []
        
        def hook(module, input, output):
            all_features.append(input[0].detach().cpu())

        handle = self.model.base_model.fc.register_forward_hook(hook)
        
        self.model.to(device)
        with torch.no_grad():
            for inputs, _ in dataloader:
                _, out = self.model(inputs.to(device))
                all_logits.append(out.cpu())
        
        handle.remove()
        return torch.cat(all_features), torch.cat(all_logits)

    def compute_stats_mahalanobis(self, features):
        """Calcule la moyenne et la matrice de précision globale."""
        mean = torch.mean(features, dim=0)
        cov = torch.cov(features.T)
        precision = torch.inverse(cov + 1e-6 * torch.eye(cov.size(0)))
        return mean, precision
    