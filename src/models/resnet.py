"""
ResNet implementation for Sapling ML
Residual network architecture for plant disease classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, List
import logging

logger = logging.getLogger(__name__)


class BasicBlock(nn.Module):
    """Basic ResNet block"""
    
    expansion = 1
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out, inplace=True)
        
        return out


class Bottleneck(nn.Module):
    """Bottleneck ResNet block"""
    
    expansion = 4
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out, inplace=True)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out, inplace=True)
        
        return out


class ResNet(nn.Module):
    """ResNet architecture for plant disease classification"""
    
    def __init__(self, 
                 block: type,
                 layers: List[int],
                 num_classes: int = 39,
                 dropout_rate: float = 0.2,
                 pretrained: bool = True):
        """
        Initialize ResNet
        
        Args:
            block: Block type (BasicBlock or Bottleneck)
            layers: Number of blocks in each layer
            num_classes: Number of output classes
            dropout_rate: Dropout rate for the classifier
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.in_channels = 64
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512 * block.expansion, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        if pretrained:
            self._load_pretrained_weights()
    
    def _make_layer(self, block: type, out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        """Create a ResNet layer"""
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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
            
            # Create pretrained model
            if self.__class__.__name__ == "ResNet18":
                pretrained_model = models.resnet18(pretrained=True)
            elif self.__class__.__name__ == "ResNet34":
                pretrained_model = models.resnet34(pretrained=True)
            elif self.__class__.__name__ == "ResNet50":
                pretrained_model = models.resnet50(pretrained=True)
            elif self.__class__.__name__ == "ResNet101":
                pretrained_model = models.resnet101(pretrained=True)
            elif self.__class__.__name__ == "ResNet152":
                pretrained_model = models.resnet152(pretrained=True)
            else:
                logger.warning("No pretrained weights available for this ResNet variant")
                return
            
            # Copy compatible weights
            state_dict = self.state_dict()
            pretrained_dict = pretrained_model.state_dict()
            
            # Filter out incompatible keys (exclude classifier)
            compatible_dict = {k: v for k, v in pretrained_dict.items() 
                             if k in state_dict and v.shape == state_dict[k].shape and 'classifier' not in k}
            
            state_dict.update(compatible_dict)
            self.load_state_dict(state_dict)
            
            logger.info(f"Loaded {len(compatible_dict)} pretrained weights from ImageNet")
            
        except Exception as e:
            logger.warning(f"Failed to load pretrained weights: {str(e)}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.maxpool(x)
        
        # ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Classifier
        x = self.classifier(x)
        
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classifier"""
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.maxpool(x)
        
        # ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x
    
    def get_conv_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the last convolutional layer"""
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.maxpool(x)
        
        # ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x


class ResNet18(ResNet):
    """ResNet-18 model"""
    
    def __init__(self, num_classes: int = 39, dropout_rate: float = 0.2, pretrained: bool = True):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes, dropout_rate, pretrained)


class ResNet34(ResNet):
    """ResNet-34 model"""
    
    def __init__(self, num_classes: int = 39, dropout_rate: float = 0.2, pretrained: bool = True):
        super().__init__(BasicBlock, [3, 4, 6, 3], num_classes, dropout_rate, pretrained)


class ResNet50(ResNet):
    """ResNet-50 model"""
    
    def __init__(self, num_classes: int = 39, dropout_rate: float = 0.2, pretrained: bool = True):
        super().__init__(Bottleneck, [3, 4, 6, 3], num_classes, dropout_rate, pretrained)


class ResNet101(ResNet):
    """ResNet-101 model"""
    
    def __init__(self, num_classes: int = 39, dropout_rate: float = 0.2, pretrained: bool = True):
        super().__init__(Bottleneck, [3, 4, 23, 3], num_classes, dropout_rate, pretrained)


class ResNet152(ResNet):
    """ResNet-152 model"""
    
    def __init__(self, num_classes: int = 39, dropout_rate: float = 0.2, pretrained: bool = True):
        super().__init__(Bottleneck, [3, 8, 36, 3], num_classes, dropout_rate, pretrained)


def create_resnet18(num_classes: int = 39, dropout_rate: float = 0.2, pretrained: bool = True) -> ResNet18:
    """Create ResNet-18 model"""
    return ResNet18(num_classes, dropout_rate, pretrained)


def create_resnet34(num_classes: int = 39, dropout_rate: float = 0.2, pretrained: bool = True) -> ResNet34:
    """Create ResNet-34 model"""
    return ResNet34(num_classes, dropout_rate, pretrained)


def create_resnet50(num_classes: int = 39, dropout_rate: float = 0.2, pretrained: bool = True) -> ResNet50:
    """Create ResNet-50 model"""
    return ResNet50(num_classes, dropout_rate, pretrained)


def create_resnet101(num_classes: int = 39, dropout_rate: float = 0.2, pretrained: bool = True) -> ResNet101:
    """Create ResNet-101 model"""
    return ResNet101(num_classes, dropout_rate, pretrained)


def create_resnet152(num_classes: int = 39, dropout_rate: float = 0.2, pretrained: bool = True) -> ResNet152:
    """Create ResNet-152 model"""
    return ResNet152(num_classes, dropout_rate, pretrained)


def main():
    """Test ResNet implementation"""
    # Create model
    model = create_resnet101(num_classes=39)
    
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
