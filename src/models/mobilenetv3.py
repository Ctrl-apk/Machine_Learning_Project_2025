"""
MobileNetV3 implementation for Sapling ML
Fucking optimized for mobile deployment with efficient architecture
Because apparently phones need to be smart too
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)


class HardSwish(nn.Module):
    """Hard Swish activation function - the hardcore version of Swish"""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * F.hardtanh(x + 3, 0, 6) / 6


class HardSigmoid(nn.Module):
    """Hard Sigmoid activation function - because regular sigmoid is too soft"""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.hardtanh(x + 3, 0, 6) / 6


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block - the attention mechanism that actually works"""
    
    def __init__(self, in_channels: int, se_channels: int):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, se_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(se_channels, in_channels, 1),
            HardSigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)


class InvertedResidual(nn.Module):
    """Inverted residual block with SE - the sexy mobile convolution block"""
    
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int,
                 stride: int,
                 expand_ratio: int,
                 se_ratio: float = 0.25,
                 activation: str = "relu"):
        super().__init__()
        
        self.stride = stride
        self.use_se = se_ratio > 0
        self.use_residual = stride == 1 and in_channels == out_channels
        
        hidden_dim = int(in_channels * expand_ratio)
        se_channels = int(in_channels * se_ratio) if self.use_se else 0
        
        # Point-wise expansion
        if expand_ratio != 1:
            self.expand_conv = nn.Conv2d(in_channels, hidden_dim, 1, bias=False)
            self.expand_bn = nn.BatchNorm2d(hidden_dim)
            self.expand_act = nn.ReLU(inplace=True) if activation == "relu" else HardSwish()
        else:
            self.expand_conv = None
        
        # Depth-wise convolution
        self.depthwise_conv = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size, stride, 
            padding=kernel_size//2, groups=hidden_dim, bias=False
        )
        self.depthwise_bn = nn.BatchNorm2d(hidden_dim)
        self.depthwise_act = nn.ReLU(inplace=True) if activation == "relu" else HardSwish()
        
        # Squeeze-and-Excitation
        if self.use_se:
            self.se = SqueezeExcitation(hidden_dim, se_channels)
        
        # Point-wise projection
        self.project_conv = nn.Conv2d(hidden_dim, out_channels, 1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # Expansion
        if self.expand_conv is not None:
            x = self.expand_conv(x)
            x = self.expand_bn(x)
            x = self.expand_act(x)
        
        # Depth-wise convolution
        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x)
        x = self.depthwise_act(x)
        
        # Squeeze-and-Excitation
        if self.use_se:
            x = self.se(x)
        
        # Projection
        x = self.project_conv(x)
        x = self.project_bn(x)
        
        # Residual connection
        if self.use_residual:
            x = x + identity
        
        return x


class MobileNetV3(nn.Module):
    """MobileNetV3 architecture for plant disease classification"""
    
    def __init__(self, 
                 num_classes: int = 39,
                 width_multiplier: float = 1.0,
                 dropout_rate: float = 0.2,
                 pretrained: bool = True):
        """
        Initialize MobileNetV3
        
        Args:
            num_classes: Number of output classes
            width_multiplier: Width multiplier for the network
            dropout_rate: Dropout rate for the classifier
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.width_multiplier = width_multiplier
        
        # First layer
        first_channels = int(16 * width_multiplier)
        self.first_conv = nn.Conv2d(3, first_channels, 3, 2, 1, bias=False)
        self.first_bn = nn.BatchNorm2d(first_channels)
        self.first_act = HardSwish()
        
        # Inverted residual blocks
        self.blocks = nn.ModuleList([
            # Block 1
            InvertedResidual(first_channels, 16, 3, 1, 1, 0, "relu"),
            
            # Block 2
            InvertedResidual(16, 24, 3, 2, 4, 0, "relu"),
            InvertedResidual(24, 24, 3, 1, 3, 0, "relu"),
            
            # Block 3
            InvertedResidual(24, 40, 3, 2, 3, 0.25, "relu"),
            InvertedResidual(40, 40, 3, 1, 3, 0.25, "relu"),
            InvertedResidual(40, 40, 3, 1, 3, 0.25, "relu"),
            
            # Block 4
            InvertedResidual(40, 80, 3, 2, 6, 0.25, "hswish"),
            InvertedResidual(80, 80, 3, 1, 2.5, 0.25, "hswish"),
            InvertedResidual(80, 80, 3, 1, 2.3, 0.25, "hswish"),
            InvertedResidual(80, 80, 3, 1, 2.3, 0.25, "hswish"),
            
            # Block 5
            InvertedResidual(80, 112, 3, 1, 6, 0.25, "hswish"),
            InvertedResidual(112, 112, 3, 1, 6, 0.25, "hswish"),
            
            # Block 6
            InvertedResidual(112, 160, 3, 2, 6, 0.25, "hswish"),
            InvertedResidual(160, 160, 3, 1, 6, 0.25, "hswish"),
            InvertedResidual(160, 160, 3, 1, 6, 0.25, "hswish"),
        ])
        
        # Last layer
        last_channels = int(960 * width_multiplier)
        self.last_conv = nn.Conv2d(160, last_channels, 1, bias=False)
        self.last_bn = nn.BatchNorm2d(last_channels)
        self.last_act = HardSwish()
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(last_channels, 1280),
            HardSwish(),
            nn.Dropout(dropout_rate),
            nn.Linear(1280, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        if pretrained:
            self._load_pretrained_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _load_pretrained_weights(self):
        """Load pretrained weights from ImageNet"""
        try:
            import torchvision.models as models
            pretrained_model = models.mobilenet_v3_large(pretrained=True)
            
            # Copy compatible weights
            state_dict = self.state_dict()
            pretrained_dict = pretrained_model.state_dict()
            
            # Filter out incompatible keys
            compatible_dict = {k: v for k, v in pretrained_dict.items() 
                             if k in state_dict and v.shape == state_dict[k].shape}
            
            state_dict.update(compatible_dict)
            self.load_state_dict(state_dict)
            
            logger.info(f"Loaded {len(compatible_dict)} pretrained weights from ImageNet")
            
        except Exception as e:
            logger.warning(f"Failed to load pretrained weights: {str(e)}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # First layer
        x = self.first_conv(x)
        x = self.first_bn(x)
        x = self.first_act(x)
        
        # Inverted residual blocks
        for block in self.blocks:
            x = block(x)
        
        # Last layer
        x = self.last_conv(x)
        x = self.last_bn(x)
        x = self.last_act(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Classifier
        x = self.classifier(x)
        
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classifier"""
        # First layer
        x = self.first_conv(x)
        x = self.first_bn(x)
        x = self.first_act(x)
        
        # Inverted residual blocks
        for block in self.blocks:
            x = block(x)
        
        # Last layer
        x = self.last_conv(x)
        x = self.last_bn(x)
        x = self.last_act(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x
    
    def get_conv_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the last convolutional layer"""
        # First layer
        x = self.first_conv(x)
        x = self.first_bn(x)
        x = self.first_act(x)
        
        # Inverted residual blocks
        for block in self.blocks:
            x = block(x)
        
        # Last layer
        x = self.last_conv(x)
        x = self.last_bn(x)
        x = self.last_act(x)
        
        return x


def create_mobilenetv3_large(num_classes: int = 39, 
                            dropout_rate: float = 0.2,
                            pretrained: bool = True) -> MobileNetV3:
    """Create MobileNetV3-Large model"""
    return MobileNetV3(
        num_classes=num_classes,
        width_multiplier=1.0,
        dropout_rate=dropout_rate,
        pretrained=pretrained
    )


def create_mobilenetv3_small(num_classes: int = 39,
                            dropout_rate: float = 0.2,
                            pretrained: bool = True) -> MobileNetV3:
    """Create MobileNetV3-Small model"""
    return MobileNetV3(
        num_classes=num_classes,
        width_multiplier=0.75,
        dropout_rate=dropout_rate,
        pretrained=pretrained
    )


def main():
    """Test MobileNetV3 implementation"""
    # Create model
    model = create_mobilenetv3_large(num_classes=39)
    
    # Test forward pass
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test feature extraction
    features = model.get_features(x)
    conv_features = model.get_conv_features(x)
    
    print(f"Feature shape: {features.shape}")
    print(f"Conv feature shape: {conv_features.shape}")


if __name__ == "__main__":
    main()
