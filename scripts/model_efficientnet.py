import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


class RoadAuditEfficientNet(nn.Module):
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        
        if pretrained:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            self.backbone = models.efficientnet_b0(weights=weights)
        else:
            self.backbone = models.efficientnet_b0(weights=None)
        
        num_features = self.backbone.classifier[1].in_features
        
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(num_features, num_classes)
        )
        
        self.num_classes = num_classes
    
    def forward(self, x):
        return self.backbone(x)
    
    def freeze_backbone(self):
        for param in self.backbone.features.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("="*60)
    print("EFFICIENTNET-B0 FOR ROAD AUDIT")
    print("="*60)
    
    model = RoadAuditEfficientNet(num_classes=2, pretrained=True)
    
    x = torch.randn(4, 3, 224, 224)
    
    print(f"\nInput shape:  {x.shape}")
    print(f"Output shape: {model(x).shape}")
    print(f"\nTrainable parameters: {count_parameters(model):,}")
    
    model.freeze_backbone()
    print(f"Trainable (frozen):   {count_parameters(model):,}")
    
    model.unfreeze_backbone()
    print(f"Trainable (unfrozen): {count_parameters(model):,}")
    
    print("\n" + "="*60)
    print("RECOMMENDED SETTINGS:")
    print("="*60)
    print("- Input size: 224×224")
    print("- Batch size: 64 (3050Ti) or 128 (Colab)")
    print("- Learning rate: 1e-4 (frozen) or 3e-5 (unfrozen)")
    print("- Epochs: 10-15 (frozen) + 5-10 (unfrozen)")
    print("="*60)
