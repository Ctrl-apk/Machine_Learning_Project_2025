"""
TensorFlow Lite export utilities for Sapling ML
Model quantization and mobile optimization - because phones need to be smart too
"""

import os
import torch
import tensorflow as tf
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging
import json
from tqdm import tqdm

logger = logging.getLogger(__name__)


class TFLiteExporter:
    """TensorFlow Lite model exporter with quantization support - the mobile export bitch"""
    
    def __init__(self, model: torch.nn.Module, device: torch.device):
        """
        Initialize TFLite exporter - because we need to make this shit mobile
        
        Args:
            model: PyTorch model to export
            device: Device to run model on
        """
        self.model = model
        self.device = device
        self.model.eval()
    
    def export_to_onnx(self, output_path: Path, input_shape: Tuple[int, ...] = (1, 3, 224, 224)) -> Path:
        """
        Export PyTorch model to ONNX format first
        
        Args:
            output_path: Path to save ONNX model
            input_shape: Input tensor shape (batch_size, channels, height, width)
            
        Returns:
            Path to the exported ONNX model
        """
        logger.info("Exporting model to ONNX format")
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        logger.info(f"ONNX model exported to {output_path}")
        return output_path
    
    def convert_onnx_to_tflite(self, onnx_path: Path, tflite_path: Path, 
                              quantization: str = "int8") -> Path:
        """
        Convert ONNX model to TensorFlow Lite
        
        Args:
            onnx_path: Path to ONNX model
            tflite_path: Path to save TFLite model
            quantization: Quantization type ('float32', 'int8', 'float16')
            
        Returns:
            Path to the exported TFLite model
        """
        logger.info(f"Converting ONNX to TFLite with {quantization} quantization")
        
        try:
            import onnx
            import tf2onnx
            
            # Load ONNX model
            onnx_model = onnx.load(str(onnx_path))
            
            # Convert to TensorFlow
            tf_model_path = tflite_path.parent / "temp_tf_model"
            tf_model_path.mkdir(exist_ok=True)
            
            # Convert ONNX to TensorFlow
            from onnx_tf.backend import prepare
            tf_rep = prepare(onnx_model)
            tf_rep.export_graph(str(tf_model_path))
            
            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_model_path))
            
            if quantization == "int8":
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
            elif quantization == "float16":
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]
            else:  # float32
                pass
            
            # Convert
            tflite_model = converter.convert()
            
            # Save
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            logger.info(f"TFLite model saved to {tflite_path}")
            
            # Clean up temporary files
            import shutil
            shutil.rmtree(tf_model_path)
            
            return tflite_path
            
        except ImportError as e:
            logger.error(f"Required packages not installed: {e}")
            logger.error("Install with: pip install onnx onnx-tf tf2onnx")
            raise
        except Exception as e:
            logger.error(f"Failed to convert ONNX to TFLite: {e}")
            raise
    
    def export_with_quantization_aware_training(self, output_path: Path, 
                                              representative_dataset: List[torch.Tensor],
                                              input_shape: Tuple[int, ...] = (1, 3, 224, 224)) -> Path:
        """
        Export model with quantization-aware training
        
        Args:
            output_path: Path to save TFLite model
            representative_dataset: Representative dataset for quantization
            input_shape: Input tensor shape
            
        Returns:
            Path to the exported TFLite model
        """
        logger.info("Exporting with quantization-aware training")
        
        # First export to ONNX
        onnx_path = output_path.parent / f"{output_path.stem}_temp.onnx"
        self.export_to_onnx(onnx_path, input_shape)
        
        # Convert to TFLite with quantization
        tflite_path = self.convert_onnx_to_tflite(onnx_path, output_path, "int8")
        
        # Clean up ONNX file
        onnx_path.unlink()
        
        return tflite_path
    
    def create_representative_dataset(self, data_loader, num_samples: int = 100) -> List[torch.Tensor]:
        """
        Create representative dataset for quantization
        
        Args:
            data_loader: DataLoader for the dataset
            num_samples: Number of samples to use
            
        Returns:
            List of representative input tensors
        """
        logger.info(f"Creating representative dataset with {num_samples} samples")
        
        representative_data = []
        count = 0
        
        with torch.no_grad():
            for images, _, _ in data_loader:
                if count >= num_samples:
                    break
                
                for i in range(images.size(0)):
                    if count >= num_samples:
                        break
                    
                    representative_data.append(images[i:i+1])
                    count += 1
        
        logger.info(f"Created representative dataset with {len(representative_data)} samples")
        return representative_data
    
    def validate_tflite_model(self, tflite_path: Path, test_input: torch.Tensor) -> Dict[str, Any]:
        """
        Validate TFLite model against PyTorch model
        
        Args:
            tflite_path: Path to TFLite model
            test_input: Test input tensor
            
        Returns:
            Validation results
        """
        logger.info("Validating TFLite model")
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Prepare input
        input_data = test_input.numpy().astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        interpreter.invoke()
        
        # Get output
        tflite_output = interpreter.get_tensor(output_details[0]['index'])
        
        # Compare with PyTorch model
        with torch.no_grad():
            pytorch_output = self.model(test_input.to(self.device)).cpu().numpy()
        
        # Calculate differences
        mse = np.mean((pytorch_output - tflite_output) ** 2)
        mae = np.mean(np.abs(pytorch_output - tflite_output))
        max_diff = np.max(np.abs(pytorch_output - tflite_output))
        
        # Check if outputs are close
        outputs_close = np.allclose(pytorch_output, tflite_output, atol=1e-3)
        
        validation_results = {
            'mse': float(mse),
            'mae': float(mae),
            'max_difference': float(max_diff),
            'outputs_close': bool(outputs_close),
            'pytorch_output': pytorch_output.tolist(),
            'tflite_output': tflite_output.tolist()
        }
        
        logger.info(f"Validation results: MSE={mse:.6f}, MAE={mae:.6f}, Max Diff={max_diff:.6f}")
        logger.info(f"Outputs close: {outputs_close}")
        
        return validation_results
    
    def get_model_info(self, tflite_path: Path) -> Dict[str, Any]:
        """Get information about the TFLite model"""
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Get model size
        model_size = tflite_path.stat().st_size
        
        info = {
            'model_path': str(tflite_path),
            'model_size_bytes': model_size,
            'model_size_mb': model_size / (1024 * 1024),
            'input_details': input_details,
            'output_details': output_details,
            'num_inputs': len(input_details),
            'num_outputs': len(output_details)
        }
        
        return info


class MobileOptimizer:
    """Mobile-specific optimizations for TFLite models"""
    
    @staticmethod
    def optimize_for_mobile(tflite_path: Path, optimized_path: Path) -> Path:
        """
        Apply mobile-specific optimizations
        
        Args:
            tflite_path: Path to input TFLite model
            optimized_path: Path to save optimized model
            
        Returns:
            Path to optimized model
        """
        logger.info("Applying mobile optimizations")
        
        # Load model
        with open(tflite_path, 'rb') as f:
            model_data = f.read()
        
        # Apply optimizations
        converter = tf.lite.TFLiteConverter.from_saved_model(str(tflite_path))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Mobile-specific optimizations
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        
        # Convert
        optimized_model = converter.convert()
        
        # Save
        with open(optimized_path, 'wb') as f:
            f.write(optimized_model)
        
        logger.info(f"Mobile-optimized model saved to {optimized_path}")
        return optimized_path
    
    @staticmethod
    def create_model_metadata(model_info: Dict[str, Any], class_names: List[str]) -> Dict[str, Any]:
        """Create metadata for the mobile model"""
        metadata = {
            'model_info': model_info,
            'class_names': class_names,
            'num_classes': len(class_names),
            'input_shape': model_info['input_details'][0]['shape'],
            'output_shape': model_info['output_details'][0]['shape'],
            'preprocessing': {
                'normalize': True,
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'input_range': [0, 1]
            },
            'postprocessing': {
                'apply_softmax': True,
                'top_k': 5
            }
        }
        
        return metadata


def main():
    """CLI interface for TFLite export"""
    import argparse
    import yaml
    from ..models.efficientnet import create_efficientnet_b0
    from ..data.dataset_loader import DataLoaderFactory, load_class_mapping
    
    parser = argparse.ArgumentParser(description="Export model to TensorFlow Lite")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--quantization", type=str, default="int8", choices=["float32", "int8", "float16"], help="Quantization type")
    parser.add_argument("--test-split", type=str, help="Path to test split CSV for representative dataset")
    parser.add_argument("--image-dir", type=str, help="Base image directory")
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
    
    # Create exporter
    exporter = TFLiteExporter(model, device)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export model
    tflite_path = output_dir / f"model_{args.quantization}.tflite"
    
    if args.test_split and args.image_dir:
        # Create representative dataset
        test_df = pd.read_csv(args.test_split)
        _, _, test_loader = DataLoaderFactory.create_data_loaders(
            config, test_df, test_df, test_df, Path(args.image_dir), class_mapping
        )
        representative_data = exporter.create_representative_dataset(test_loader, 100)
        
        # Export with quantization
        exporter.export_with_quantization_aware_training(tflite_path, representative_data)
    else:
        # Simple export
        onnx_path = output_dir / "model.onnx"
        exporter.export_to_onnx(onnx_path)
        exporter.convert_onnx_to_tflite(onnx_path, tflite_path, args.quantization)
        onnx_path.unlink()
    
    # Get model info
    model_info = exporter.get_model_info(tflite_path)
    
    # Create metadata
    metadata = MobileOptimizer.create_model_metadata(model_info, class_names)
    
    # Save metadata
    with open(output_dir / "model_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Validate model
    test_input = torch.randn(1, 3, 224, 224)
    validation_results = exporter.validate_tflite_model(tflite_path, test_input)
    
    # Save validation results
    with open(output_dir / "validation_results.json", 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    print(f"TFLite model exported to {tflite_path}")
    print(f"Model size: {model_info['model_size_mb']:.2f} MB")
    print(f"Validation: Outputs close = {validation_results['outputs_close']}")


if __name__ == "__main__":
    main()
