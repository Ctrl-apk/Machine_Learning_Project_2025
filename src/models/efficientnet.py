"""
EfficientNet implementation for Sapling ML
Fucking efficient and scalable architecture for plant disease classification
Because apparently we need to be fancy with our model choices
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, List
import math
import logging

logger = logging.getLogger(__name__)


class Swish(nn.Module):
    """Swish activation function - the sexy smooth alternative to ReLU"""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block - because attention is fucking important"""
    
    def __init__(self, in_channels: int, se_ratio: float = 0.25):
        super().__init__()
        se_channels = max(1, int(in_channels * se_ratio))
        
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, se_channels, 1),
            Swish(),
            nn.Conv2d(se_channels, in_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)


class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Convolution block - the bitch that makes everything work"""
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 expand_ratio: int,
                 se_ratio: float = 0.25,
                 dropout_rate: float = 0.0):
        super().__init__()
        
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        # Expansion phase - make this shit bigger
        expanded_channels = in_channels * expand_ratio
        if expand_ratio != 1:
            self.expand_conv = nn.Conv2d(in_channels, expanded_channels, 1, bias=False)
            self.expand_bn = nn.BatchNorm2d(expanded_channels)
            self.expand_swish = Swish()
        else:
            self.expand_conv = None
        
        # Depthwise convolution - the sexy part that does the heavy lifting
        self.depthwise_conv = nn.Conv2d(
            expanded_channels, expanded_channels, kernel_size, stride,
            padding=kernel_size//2, groups=expanded_channels, bias=False
        )
        self.depthwise_bn = nn.BatchNorm2d(expanded_channels)
        self.depthwise_swish = Swish()
        
        # Squeeze-and-Excitation - because we need attention
        self.se = SqueezeExcitation(expanded_channels, se_ratio)
        
        # Projection phase - squeeze it back down
        self.project_conv = nn.Conv2d(expanded_channels, out_channels, 1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)
        
        # Dropout - the fucking regularization that prevents overfitting
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # Expansion - blow this bitch up
        if self.expand_conv is not None:
            x = self.expand_conv(x)
            x = self.expand_bn(x)
            x = self.expand_swish(x)
        
        # Depthwise convolution - the meat and potatoes
        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x)
        x = self.depthwise_swish(x)
        
        # Squeeze-and-Excitation - pay attention, bitch
        x = self.se(x)
        
        # Projection - squeeze it back down to size
        x = self.project_conv(x)
        x = self.project_bn(x)
        
        # Dropout - randomly kill some neurons because fuck overfitting
        if self.dropout is not None:
            x = self.dropout(x)
        
        # Residual connection - the sexy shortcut that makes everything work
        if self.use_residual:
            x = x + identity
        
        return x


class EfficientNet(nn.Module):
    """EfficientNet architecture for plant disease classification"""
    
    def __init__(self, 
                 num_classes: int = 39,
                 width_coefficient: float = 1.0,
                 depth_coefficient: float = 1.0,
                 dropout_rate: float = 0.2,
                 drop_connect_rate: float = 0.2,
                 pretrained: bool = True):
        """
        Initialize EfficientNet
        
        Args:
            num_classes: Number of output classes
            width_coefficient: Width scaling coefficient
            depth_coefficient: Depth scaling coefficient
            dropout_rate: Dropout rate for the classifier
            drop_connect_rate: Dropout rate for MBConv blocks
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        
        # Calculate scaled dimensions
        def round_filters(filters):
            multiplier = width_coefficient
            divisor = 8
            min_depth = None
            if multiplier < 1.0:
                min_depth = divisor
            filters = int(filters * multiplier)
            if min_depth is not None:
                filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
            return int(filters)
        
        def round_repeats(repeats):
            return int(math.ceil(depth_coefficient * repeats))
        
        # Stem
        stem_channels = round_filters(32)
        self.stem_conv = nn.Conv2d(3, stem_channels, 3, 2, 1, bias=False)
        self.stem_bn = nn.BatchNorm2d(stem_channels)
        self.stem_swish = Swish()
        
        # MBConv blocks configuration
        blocks_config = [
            # (in_channels, out_channels, kernel_size, stride, expand_ratio, num_repeats, se_ratio)
            (32, 16, 3, 1, 1, 1, 0.25),      # MBConv1
            (16, 24, 3, 2, 6, 2, 0.25),      # MBConv2
            (24, 40, 5, 2, 6, 2, 0.25),      # MBConv3
            (40, 80, 3, 2, 6, 3, 0.25),      # MBConv4
            (80, 112, 5, 1, 6, 3, 0.25),     # MBConv5
            (112, 192, 5, 2, 6, 4, 0.25),    # MBConv6
            (192, 320, 3, 1, 6, 1, 0.25),    # MBConv7
        ]
        
        # Build MBConv blocks
        self.blocks = nn.ModuleList()
        in_channels = stem_channels
        
        for i, (in_ch, out_ch, kernel_size, stride, expand_ratio, num_repeats, se_ratio) in enumerate(blocks_config):
            in_ch = round_filters(in_ch)
            out_ch = round_filters(out_ch)
            num_repeats = round_repeats(num_repeats)
            
            # First block
            block = MBConvBlock(
                in_channels, out_ch, kernel_size, stride, expand_ratio, se_ratio, drop_connect_rate
            )
            self.blocks.append(block)
            in_channels = out_ch
            
            # Additional blocks
            for _ in range(num_repeats - 1):
                block = MBConvBlock(
                    in_channels, out_ch, kernel_size, 1, expand_ratio, se_ratio, drop_connect_rate
                )
                self.blocks.append(block)
        
        # Head
        head_channels = round_filters(1280)
        self.head_conv = nn.Conv2d(in_channels, head_channels, 1, bias=False)
        self.head_bn = nn.BatchNorm2d(head_channels)
        self.head_swish = Swish()
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(head_channels, num_classes)
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
            
            # Try to load from torchvision
            if hasattr(models, 'efficientnet_b0'):
                pretrained_model = models.efficientnet_b0(pretrained=True)
            else:
                # Fallback to timm
                import timm
                pretrained_model = timm.create_model('efficientnet_b0', pretrained=True)
            
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
        # Stem
        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = self.stem_swish(x)
        
        # MBConv blocks
        for block in self.blocks:
            x = block(x)
        
        # Head
        x = self.head_conv(x)
        x = self.head_bn(x)
        x = self.head_swish(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Classifier
        x = self.classifier(x)
        
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classifier"""
        # Stem
        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = self.stem_swish(x)
        
        # MBConv blocks
        for block in self.blocks:
            x = block(x)
        
        # Head
        x = self.head_conv(x)
        x = self.head_bn(x)
        x = self.head_swish(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x
    
    def get_conv_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the last convolutional layer"""
        # Stem
        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = self.stem_swish(x)
        
        # MBConv blocks
        for block in self.blocks:
            x = block(x)
        
        # Head
        x = self.head_conv(x)
        x = self.head_bn(x)
        x = self.head_swish(x)
        
        return x


def create_efficientnet_b0(num_classes: int = 39, 
                          dropout_rate: float = 0.2,
                          pretrained: bool = True) -> EfficientNet:
    """Create EfficientNet-B0 model"""
    return EfficientNet(
        num_classes=num_classes,
        width_coefficient=1.0,
        depth_coefficient=1.0,
        dropout_rate=dropout_rate,
        pretrained=pretrained
    )


def create_efficientnet_b1(num_classes: int = 39,
                          dropout_rate: float = 0.2,
                          pretrained: bool = True) -> EfficientNet:
    """Create EfficientNet-B1 model"""
    return EfficientNet(
        num_classes=num_classes,
        width_coefficient=1.0,
        depth_coefficient=1.1,
        dropout_rate=dropout_rate,
        pretrained=pretrained
    )


def create_efficientnet_b2(num_classes: int = 39,
                          dropout_rate: float = 0.3,
                          pretrained: bool = True) -> EfficientNet:
    """Create EfficientNet-B2 model"""
    return EfficientNet(
        num_classes=num_classes,
        width_coefficient=1.1,
        depth_coefficient=1.2,
        dropout_rate=dropout_rate,
        pretrained=pretrained
    )


def create_efficientnet_b3(num_classes: int = 39,
                          dropout_rate: float = 0.3,
                          pretrained: bool = True) -> EfficientNet:
    """Create EfficientNet-B3 model"""
    return EfficientNet(
        num_classes=num_classes,
        width_coefficient=1.2,
        depth_coefficient=1.4,
        dropout_rate=dropout_rate,
        pretrained=pretrained
    )


def main():
    """Test EfficientNet implementation"""
    # Create model
    model = create_efficientnet_b0(num_classes=39)
    
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
