"""
Grad-CAM implementation for Sapling ML
Explainable AI for plant disease classification - because we need to know why this bitch made that decision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class GradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM) - the sexy explainability method"""
    
    def __init__(self, model: nn.Module, target_layers: List[str]):
        """
        Initialize Grad-CAM - because we need to see what the fuck the model is looking at
        
        Args:
            model: PyTorch model
            target_layers: List of layer names to generate Grad-CAM for
        """
        self.model = model
        self.target_layers = target_layers
        self.gradients = {}
        self.activations = {}
        self.hooks = []
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks - because we need to spy on this bitch"""
        def get_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        
        def get_gradient(name):
            def hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0].detach()
            return hook
        
        # Register hooks for target layers - spy on the important parts
        for name, module in self.model.named_modules():
            if name in self.target_layers:
                self.hooks.append(module.register_forward_hook(get_activation(name)))
                self.hooks.append(module.register_backward_hook(get_gradient(name)))
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def generate_cam(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Generate Grad-CAM for the input
        
        Args:
            input_tensor: Input tensor (batch_size, channels, height, width)
            class_idx: Class index to generate CAM for. If None, uses predicted class
            
        Returns:
            Dictionary mapping layer names to CAM arrays
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Forward pass
        input_tensor.requires_grad_(True)
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Generate CAM for each target layer
        cams = {}
        for layer_name in self.target_layers:
            if layer_name in self.activations and layer_name in self.gradients:
                cam = self._compute_cam(layer_name)
                cams[layer_name] = cam
            else:
                logger.warning(f"Layer {layer_name} not found in activations or gradients")
        
        return cams
    
    def _compute_cam(self, layer_name: str) -> np.ndarray:
        """Compute CAM for a specific layer"""
        # Get gradients and activations
        gradients = self.gradients[layer_name]  # (batch_size, channels, height, width)
        activations = self.activations[layer_name]  # (batch_size, channels, height, width)
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (batch_size, channels, 1, 1)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # (batch_size, 1, height, width)
        
        # Apply ReLU to get positive activations only
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.ndim == 2:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        else:
            # Handle batch dimension
            cam = np.array([(c - c.min()) / (c.max() - c.min() + 1e-8) for c in cam])
        
        return cam
    
    def visualize_cam(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None,
                     save_path: Optional[Path] = None, figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:
        """
        Visualize Grad-CAM results
        
        Args:
            input_tensor: Input tensor
            class_idx: Class index to visualize
            save_path: Path to save the visualization
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Generate CAM
        cams = self.generate_cam(input_tensor, class_idx)
        
        # Get original image
        original_image = input_tensor.squeeze().cpu().numpy()
        if original_image.shape[0] == 3:  # CHW format
            original_image = np.transpose(original_image, (1, 2, 0))
        
        # Normalize image to [0, 1]
        original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())
        
        # Create subplots
        num_layers = len(cams)
        fig, axes = plt.subplots(2, num_layers + 1, figsize=figsize)
        
        # Original image
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # CAM visualizations
        for i, (layer_name, cam) in enumerate(cams.items()):
            # Raw CAM
            im1 = axes[0, i + 1].imshow(cam, cmap='jet')
            axes[0, i + 1].set_title(f'Grad-CAM ({layer_name})')
            axes[0, i + 1].axis('off')
            plt.colorbar(im1, ax=axes[0, i + 1])
            
            # Overlay on original image
            overlay = self._overlay_cam_on_image(original_image, cam)
            axes[1, i + 1].imshow(overlay)
            axes[1, i + 1].set_title(f'Overlay ({layer_name})')
            axes[1, i + 1].axis('off')
        
        # Hide unused subplot
        axes[1, 0].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Grad-CAM visualization saved to {save_path}")
        
        return fig
    
    def _overlay_cam_on_image(self, image: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """Overlay CAM on original image"""
        # Resize CAM to match image size
        cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlay = alpha * heatmap + (1 - alpha) * image
        overlay = np.clip(overlay, 0, 1)
        
        return overlay


class GradCAMPlusPlus(GradCAM):
    """Grad-CAM++ implementation with improved localization"""
    
    def _compute_cam(self, layer_name: str) -> np.ndarray:
        """Compute Grad-CAM++ for a specific layer"""
        # Get gradients and activations
        gradients = self.gradients[layer_name]
        activations = self.activations[layer_name]
        
        # Compute alpha weights
        alpha = torch.sum(gradients, dim=(2, 3), keepdim=True)  # (batch_size, channels, 1, 1)
        alpha = F.relu(alpha)
        
        # Compute weights
        weights = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.squeeze().cpu().numpy()
        if cam.ndim == 2:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        else:
            cam = np.array([(c - c.min()) / (c.max() - c.min() + 1e-8) for c in cam])
        
        return cam


class GuidedGradCAM:
    """Guided Grad-CAM combining guided backpropagation and Grad-CAM"""
    
    def __init__(self, model: nn.Module, target_layers: List[str]):
        self.model = model
        self.target_layers = target_layers
        self.gradcam = GradCAM(model, target_layers)
        
        # Replace ReLU with guided ReLU
        self._replace_relu_with_guided_relu()
    
    def _replace_relu_with_guided_relu(self):
        """Replace ReLU layers with guided ReLU for guided backpropagation"""
        def guided_relu_hook(module, grad_input, grad_output):
            return (F.relu(grad_input[0]),)
        
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                module.register_backward_hook(guided_relu_hook)
    
    def generate_guided_cam(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Generate guided Grad-CAM"""
        # Generate regular Grad-CAM
        cams = self.gradcam.generate_cam(input_tensor, class_idx)
        
        # Generate guided gradients
        guided_grads = self._compute_guided_gradients(input_tensor, class_idx)
        
        # Combine with CAM
        guided_cams = {}
        for layer_name, cam in cams.items():
            if layer_name in guided_grads:
                guided_cam = cam * guided_grads[layer_name]
                guided_cams[layer_name] = guided_cam
            else:
                guided_cams[layer_name] = cam
        
        return guided_cams
    
    def _compute_guided_gradients(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Compute guided gradients"""
        input_tensor.requires_grad_(True)
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get guided gradients
        guided_grads = {}
        for name, module in self.model.named_modules():
            if name in self.target_layers and hasattr(module, 'weight'):
                if module.weight.grad is not None:
                    guided_grads[name] = module.weight.grad.detach().cpu().numpy()
        
        return guided_grads


class ExplainabilityAnalyzer:
    """Comprehensive explainability analysis"""
    
    def __init__(self, model: nn.Module, class_names: List[str], target_layers: List[str]):
        self.model = model
        self.class_names = class_names
        self.target_layers = target_layers
        self.gradcam = GradCAM(model, target_layers)
        self.gradcam_plus = GradCAMPlusPlus(model, target_layers)
        self.guided_gradcam = GuidedGradCAM(model, target_layers)
    
    def analyze_prediction(self, input_tensor: torch.Tensor, true_class: Optional[int] = None) -> Dict[str, Any]:
        """Analyze prediction with multiple explainability methods"""
        # Get prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = output.argmax(dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        # Generate explanations
        gradcam_results = self.gradcam.generate_cam(input_tensor, predicted_class)
        gradcam_plus_results = self.gradcam_plus.generate_cam(input_tensor, predicted_class)
        guided_gradcam_results = self.guided_gradcam.generate_guided_cam(input_tensor, predicted_class)
        
        # Top-k predictions
        top_k = 5
        top_k_indices = torch.topk(probabilities, top_k).indices[0].cpu().numpy()
        top_k_probs = torch.topk(probabilities, top_k).values[0].cpu().numpy()
        
        analysis = {
            'predicted_class': predicted_class,
            'predicted_class_name': self.class_names[predicted_class],
            'confidence': confidence,
            'true_class': true_class,
            'true_class_name': self.class_names[true_class] if true_class is not None else None,
            'correct': predicted_class == true_class if true_class is not None else None,
            'top_k_predictions': [
                {'class': int(idx), 'class_name': self.class_names[idx], 'probability': float(prob)}
                for idx, prob in zip(top_k_indices, top_k_probs)
            ],
            'gradcam': gradcam_results,
            'gradcam_plus': gradcam_plus_results,
            'guided_gradcam': guided_gradcam_results
        }
        
        return analysis
    
    def batch_analyze(self, data_loader, num_samples: int = 10) -> List[Dict[str, Any]]:
        """Analyze a batch of samples"""
        analyses = []
        
        for i, (images, labels, metadata) in enumerate(data_loader):
            if i >= num_samples:
                break
            
            for j in range(images.size(0)):
                if len(analyses) >= num_samples:
                    break
                
                single_image = images[j:j+1]
                single_label = labels[j].item()
                
                analysis = self.analyze_prediction(single_image, single_label)
                analysis['metadata'] = {
                    'filename': metadata['filename'][j],
                    'orig_id': metadata['orig_id'][j],
                    'source': metadata['source'][j]
                }
                
                analyses.append(analysis)
        
        return analyses
    
    def save_analysis_results(self, analyses: List[Dict[str, Any]], save_dir: Path):
        """Save analysis results"""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        import json
        with open(save_dir / "explainability_analysis.json", 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_analyses = []
            for analysis in analyses:
                serializable_analysis = analysis.copy()
                # Remove numpy arrays from serializable version
                for key in ['gradcam', 'gradcam_plus', 'guided_gradcam']:
                    if key in serializable_analysis:
                        del serializable_analysis[key]
                serializable_analyses.append(serializable_analysis)
            
            json.dump(serializable_analyses, f, indent=2)
        
        # Save summary statistics
        summary = self._generate_summary(analyses)
        with open(save_dir / "explainability_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Explainability analysis saved to {save_dir}")
    
    def _generate_summary(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics"""
        total_samples = len(analyses)
        correct_predictions = sum(1 for a in analyses if a.get('correct', False))
        
        confidence_scores = [a['confidence'] for a in analyses]
        
        summary = {
            'total_samples': total_samples,
            'correct_predictions': correct_predictions,
            'accuracy': correct_predictions / total_samples if total_samples > 0 else 0,
            'mean_confidence': np.mean(confidence_scores),
            'std_confidence': np.std(confidence_scores),
            'min_confidence': np.min(confidence_scores),
            'max_confidence': np.max(confidence_scores)
        }
        
        return summary


def main():
    """CLI interface for explainability analysis"""
    import argparse
    import yaml
    from ..models.efficientnet import create_efficientnet_b0
    from ..data.dataset_loader import DataLoaderFactory, load_class_mapping
    
    parser = argparse.ArgumentParser(description="Generate explainability analysis")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--test-split", type=str, required=True, help="Path to test split CSV")
    parser.add_argument("--image-dir", type=str, required=True, help="Base image directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to analyze")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    # Load class mapping
    class_mapping = load_class_mapping(Path(args.config))
    class_names = [name for name, _ in sorted(class_mapping.items(), key=lambda x: x[1])]
    
    # Create model
    model = create_efficientnet_b0(num_classes=len(class_names))
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    
    # Load test data
    test_df = pd.read_csv(args.test_split)
    _, _, test_loader = DataLoaderFactory.create_data_loaders(
        config, test_df, test_df, test_df, Path(args.image_dir), class_mapping
    )
    
    # Create analyzer
    target_layers = config.get('explainability', {}).get('gradcam', {}).get('target_layers', ['last_conv'])
    analyzer = ExplainabilityAnalyzer(model, class_names, target_layers)
    
    # Run analysis
    analyses = analyzer.batch_analyze(test_loader, args.num_samples)
    
    # Save results
    output_dir = Path(args.output_dir)
    analyzer.save_analysis_results(analyses, output_dir)
    
    print(f"Explainability analysis completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
