"""
model.py

EfficientNet-B0 with a custom classification head for interior design style prediction.
Uses timm for the pretrained backbone.

Transfer learning strategy:
  - Phase 1 (epochs 1-5):  Backbone frozen, only train classifier head
  - Phase 2 (epochs 6+):   Unfreeze top backbone layers, fine-tune at lower LR
"""

import torch
import torch.nn as nn
import timm


STYLE_CLASSES = [
    "Bohemian",
    "Industrial",
    "Mid-Century Modern",
    "Minimalist",
    "Scandinavian",
]

NUM_CLASSES = len(STYLE_CLASSES)


class InteriorStyleClassifier(nn.Module):
    """
    EfficientNet-B0 backbone with a custom head for interior style classification.

    Args:
        num_classes: Number of output classes (default: 5)
        dropout: Dropout rate before the final linear layer (default: 0.3)
        pretrained: Load ImageNet pretrained weights (default: True)
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.3,
        pretrained: bool = True,
    ):
        super().__init__()

        # Load pretrained EfficientNet-B0, strip original head
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,  # removes classifier, returns features
        )

        in_features = self.backbone.num_features  # 1280 for efficientnet_b0

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    def freeze_backbone(self):
        """Freeze all backbone parameters (Phase 1 training)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_top_layers(self, num_blocks: int = 3):
        """
        Unfreeze the top N blocks of the backbone for fine-tuning (Phase 2).
        EfficientNet-B0 has blocks named 'blocks.0' through 'blocks.6'.
        """
        # Always keep classifier trainable
        for param in self.classifier.parameters():
            param.requires_grad = True

        # Unfreeze last num_blocks blocks + head components
        blocks = list(self.backbone.blocks.children())
        for block in blocks[-num_blocks:]:
            for param in block.parameters():
                param.requires_grad = True

        # Unfreeze the conv head
        for param in self.backbone.conv_head.parameters():
            param.requires_grad = True
        for param in self.backbone.bn2.parameters():
            param.requires_grad = True

    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def load_model(weights_path: str, device: str = "cpu") -> InteriorStyleClassifier:
    """Load a trained model from a checkpoint file."""
    model = InteriorStyleClassifier(pretrained=False)
    state = torch.load(weights_path, map_location=device)

    # Handle both raw state_dict and checkpoint dicts
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)

    model.eval()
    return model.to(device)


if __name__ == "__main__":
    # Quick sanity check
    model = InteriorStyleClassifier(pretrained=False)
    print(f"Total params:     {model.total_params():,}")
    print(f"Trainable params: {model.trainable_params():,}")

    model.freeze_backbone()
    print(f"\nAfter freeze_backbone:")
    print(f"Trainable params: {model.trainable_params():,}")

    model.unfreeze_top_layers(3)
    print(f"\nAfter unfreeze_top_layers(3):")
    print(f"Trainable params: {model.trainable_params():,}")

    # Forward pass test
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"\nOutput shape: {out.shape}")  # should be [2, 5]
