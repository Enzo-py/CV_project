import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

def get_resnet18_cifar100(num_classes=100):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, num_classes)
    return model

def get_model():
    return ResNetWithFeatures(get_resnet18_cifar100())

class ResNetWithFeatures(nn.Module): # we want to return the features before the last fc
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)
        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)
        x = self.base_model.avgpool(x)
        features = torch.flatten(x, 1)
        logits = self.base_model.fc(features)
        return features, logits
    
class GeometricLoss(nn.Module):
    def __init__(self, temperature=0.1, lambda_intra=1.0, lambda_inter=2.0):
        super(GeometricLoss, self).__init__()
        self.temperature = temperature
        self.lambda_intra = lambda_intra
        self.lambda_inter = lambda_inter

    def forward(self, features, labels):
        device = features.device
        features = F.normalize(features, p=2, dim=1)
        
        one_hot = F.one_hot(labels).float() 
        count_per_class = one_hot.sum(0).unsqueeze(1)
        batch_means = torch.matmul(one_hot.T, features) / (count_per_class + 1e-8)
        
        present_mask = (count_per_class.squeeze() > 0)
        valid_means = batch_means[present_mask]
        valid_means = F.normalize(valid_means, p=2, dim=1)

        target_means = valid_means[torch.searchsorted(torch.unique(labels), labels)]
        intra_sim = torch.sum(features * target_means, dim=1)
        loss_intra = -torch.log(torch.sigmoid(intra_sim / self.temperature)).mean()

        loss_inter = torch.tensor(0.0, device=device)
        if valid_means.shape[0] > 1:
            mean_sim_matrix = torch.matmul(valid_means, valid_means.T)
            
            n_c = valid_means.shape[0]
            off_diag_mask = ~torch.eye(n_c, dtype=torch.bool, device=device)
            
            loss_inter = mean_sim_matrix[off_diag_mask].pow(2).mean()
            
            loss_center = torch.norm(valid_means.mean(dim=0))
            loss_inter += loss_center

        return self.lambda_intra * loss_intra + self.lambda_inter * loss_inter
    