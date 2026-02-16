import torch

def score_msp(logits):
    """
    Maximum Softmax Probability (MSP).
    
    Formula:
        S(x) = max_c [ exp(z_c) / sum_j exp(z_j) ]
    """
    probs = torch.softmax(logits, dim=1)
    return torch.max(probs, dim=1)[0]

def score_max_logit(logits):
    """
    Maximum Logit Score (MLS).
    
    Formula:
        S(x) = max_c (z_c)
    """
    return torch.max(logits, dim=1)[0]

def score_energy(logits, temperature=1.0):
    """
    Energy-based Score (Negative Energy for ID scoring).
    
    Formula:
        S(x) = T * log( sum_j exp(z_j / T) )
    """
    return temperature * torch.logsumexp(logits / temperature, dim=1)

def score_mahalanobis(features, mean, precision):
    """
    Mahalanobis Distance-based score.
    
    Formula:
        S(x) = - (f - mu)^T * Sigma^{-1} * (f - mu)
    """
    diff = features - mean
    term = torch.mm(diff, precision)
    score = -torch.sum(term * diff, dim=1)
    return score

def score_vim(features, logits, w, b, u, sigma_inv):
    """
    Visual Interaction Matrix (ViM) score.
    
    Formula:
        v_proj = (f - u) - W^T * W * (f - u)
        S(x) = log( sum_j exp(z_j) ) - ||v_proj||_2
    """
    features_centered = features - u
    v_proj = features_centered - (features_centered @ w.T @ w)
    residual_norm = torch.norm(v_proj, dim=1)
    
    energy = torch.logsumexp(logits, dim=1)
    
    return energy - residual_norm

class OODEvaluator:
    """
    Evaluator for extracting features and computing OOD statistics.
    """
    def __init__(self, model):
        self.model = model.eval()
        self.feature_dim = model.base_model.fc.in_features

    def get_features_and_logits(self, dataloader, device):
        """
        Extracts latent features and logits from the dataset.
        """
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
        """
        Computes the empirical mean and Tikhonov-regularized precision matrix.
        
        Formula:
            mu = 1/N * sum(f_i)
            Sigma^{-1} = (Cov(f) + epsilon * I)^{-1}
        """
        mean = torch.mean(features, dim=0)
        cov = torch.cov(features.T)
        precision = torch.inverse(cov + 1e-6 * torch.eye(cov.size(0)))
        return mean, precision