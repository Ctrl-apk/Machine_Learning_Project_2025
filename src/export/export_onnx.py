"""
ONNX export utilities for Sapling ML
Cross-platform model deployment - because we need to run this shit everywhere
"""

import os
import torch
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging
import json

logger = logging.getLogger(__name__)


class ONNXExporter:
    """ONNX model exporter with optimization support - the cross-platform export bitch"""
    
    def __init__(self, model: torch.nn.Module, device: torch.device):
        """
        Initialize ONNX exporter - because we need to make this shit universal
        
        Args:
            model: PyTorch model to export
            device: Device to run model on
        """
        self.model = model
        self.device = device
        self.model.eval()
    
    def export_model(self, output_path: Path, 
                    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
                    opset_version: int = 11,
                    dynamic_axes: bool = True) -> Path:
        """
        Export PyTorch model to ONNX format
        
        Args:
            output_path: Path to save ONNX model
            input_shape: Input tensor shape (batch_size, channels, height, width)
            opset_version: ONNX opset version
            dynamic_axes: Whether to use dynamic axes for batch size
            
        Returns:
            Path to the exported ONNX model
        """
        logger.info(f"Exporting model to ONNX format (opset {opset_version})")
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Prepare dynamic axes
        if dynamic_axes:
            dynamic_axes_dict = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        else:
            dynamic_axes_dict = None
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes_dict,
            verbose=False
        )
        
        logger.info(f"ONNX model exported to {output_path}")
        return output_path
    
    def optimize_model(self, onnx_path: Path, optimized_path: Path) -> Path:
        """
        Optimize ONNX model
        
        Args:
            onnx_path: Path to input ONNX model
            optimized_path: Path to save optimized model
            
        Returns:
            Path to optimized model
        """
        logger.info("Optimizing ONNX model")
        
        try:
            from onnxruntime.tools import optimizer
            
            # Load model
            model = onnx.load(str(onnx_path))
            
            # Apply optimizations
            optimized_model = optimizer.optimize_model(
                str(onnx_path),
                model_type='bert',  # Use general optimization
                num_heads=0,
                hidden_size=0
            )
            
            # Save optimized model
            optimized_model.save_model_to_file(str(optimized_path))
            
            logger.info(f"Optimized ONNX model saved to {optimized_path}")
            return optimized_path
            
        except ImportError:
            logger.warning("ONNX optimizer not available, skipping optimization")
            return onnx_path
        except Exception as e:
            logger.warning(f"Optimization failed: {e}, using original model")
            return onnx_path
    
    def quantize_model(self, onnx_path: Path, quantized_path: Path, 
                      quantization_type: str = "int8") -> Path:
        """
        Quantize ONNX model
        
        Args:
            onnx_path: Path to input ONNX model
            quantized_path: Path to save quantized model
            quantization_type: Quantization type ('int8', 'uint8')
            
        Returns:
            Path to quantized model
        """
        logger.info(f"Quantizing ONNX model with {quantization_type}")
        
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            
            # Set quantization type
            if quantization_type == "int8":
                quant_type = QuantType.QInt8
            elif quantization_type == "uint8":
                quant_type = QuantType.QUInt8
            else:
                raise ValueError(f"Unsupported quantization type: {quantization_type}")
            
            # Quantize model
            quantize_dynamic(
                str(onnx_path),
                str(quantized_path),
                weight_type=quant_type
            )
            
            logger.info(f"Quantized ONNX model saved to {quantized_path}")
            return quantized_path
            
        except ImportError:
            logger.warning("ONNX quantization not available, skipping quantization")
            return onnx_path
        except Exception as e:
            logger.warning(f"Quantization failed: {e}, using original model")
            return onnx_path
    
    def validate_model(self, onnx_path: Path, test_input: torch.Tensor) -> Dict[str, Any]:
        """
        Validate ONNX model against PyTorch model
        
        Args:
            onnx_path: Path to ONNX model
            test_input: Test input tensor
            
        Returns:
            Validation results
        """
        logger.info("Validating ONNX model")
        
        # Load ONNX model
        onnx_model = onnx.load(str(onnx_path))
        
        # Check model
        try:
            onnx.checker.check_model(onnx_model)
            model_valid = True
        except Exception as e:
            logger.error(f"ONNX model validation failed: {e}")
            model_valid = False
        
        # Test inference
        try:
            # Create ONNX Runtime session
            session = ort.InferenceSession(str(onnx_path))
            
            # Prepare input
            input_name = session.get_inputs()[0].name
            input_data = test_input.numpy().astype(np.float32)
            
            # Run inference
            onnx_output = session.run(None, {input_name: input_data})[0]
            
            # Compare with PyTorch model
            with torch.no_grad():
                pytorch_output = self.model(test_input.to(self.device)).cpu().numpy()
            
            # Calculate differences
            mse = np.mean((pytorch_output - onnx_output) ** 2)
            mae = np.mean(np.abs(pytorch_output - onnx_output))
            max_diff = np.max(np.abs(pytorch_output - onnx_output))
            
            # Check if outputs are close
            outputs_close = np.allclose(pytorch_output, onnx_output, atol=1e-5)
            
            inference_success = True
            
        except Exception as e:
            logger.error(f"ONNX inference failed: {e}")
            mse = mae = max_diff = float('inf')
            outputs_close = False
            inference_success = False
            onnx_output = None
            pytorch_output = None
        
        validation_results = {
            'model_valid': model_valid,
            'inference_success': inference_success,
            'mse': float(mse),
            'mae': float(mae),
            'max_difference': float(max_diff),
            'outputs_close': bool(outputs_close),
            'pytorch_output': pytorch_output.tolist() if pytorch_output is not None else None,
            'onnx_output': onnx_output.tolist() if onnx_output is not None else None
        }
        
        logger.info(f"Validation results: Model valid={model_valid}, "
                   f"Inference success={inference_success}, "
                   f"MSE={mse:.6f}, MAE={mae:.6f}, Max Diff={max_diff:.6f}")
        
        return validation_results
    
    def get_model_info(self, onnx_path: Path) -> Dict[str, Any]:
        """Get information about the ONNX model"""
        try:
            # Load model
            model = onnx.load(str(onnx_path))
            
            # Get model size
            model_size = onnx_path.stat().st_size
            
            # Get input/output info
            input_info = []
            output_info = []
            
            for input_tensor in model.graph.input:
                input_info.append({
                    'name': input_tensor.name,
                    'type': str(input_tensor.type.tensor_type.elem_type),
                    'shape': [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
                })
            
            for output_tensor in model.graph.output:
                output_info.append({
                    'name': output_tensor.name,
                    'type': str(output_tensor.type.tensor_type.elem_type),
                    'shape': [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
                })
            
            # Get opset version
            opset_version = None
            for opset in model.opset_import:
                if opset.domain == '':
                    opset_version = opset.version
                    break
            
            info = {
                'model_path': str(onnx_path),
                'model_size_bytes': model_size,
                'model_size_mb': model_size / (1024 * 1024),
                'opset_version': opset_version,
                'input_info': input_info,
                'output_info': output_info,
                'num_inputs': len(input_info),
                'num_outputs': len(output_info)
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {
                'model_path': str(onnx_path),
                'model_size_bytes': onnx_path.stat().st_size,
                'model_size_mb': onnx_path.stat().st_size / (1024 * 1024),
                'error': str(e)
            }
    
    def benchmark_model(self, onnx_path: Path, input_shape: Tuple[int, ...] = (1, 3, 224, 224),
                       num_runs: int = 100) -> Dict[str, Any]:
        """
        Benchmark ONNX model performance
        
        Args:
            onnx_path: Path to ONNX model
            input_shape: Input tensor shape
            num_runs: Number of benchmark runs
            
        Returns:
            Benchmark results
        """
        logger.info(f"Benchmarking ONNX model with {num_runs} runs")
        
        try:
            # Create session
            session = ort.InferenceSession(str(onnx_path))
            
            # Prepare input
            input_name = session.get_inputs()[0].name
            input_data = np.random.randn(*input_shape).astype(np.float32)
            
            # Warmup
            for _ in range(10):
                session.run(None, {input_name: input_data})
            
            # Benchmark
            import time
            times = []
            
            for _ in range(num_runs):
                start_time = time.time()
                session.run(None, {input_name: input_data})
                end_time = time.time()
                times.append(end_time - start_time)
            
            # Calculate statistics
            times = np.array(times)
            mean_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)
            
            benchmark_results = {
                'mean_inference_time_ms': mean_time * 1000,
                'std_inference_time_ms': std_time * 1000,
                'min_inference_time_ms': min_time * 1000,
                'max_inference_time_ms': max_time * 1000,
                'throughput_fps': 1.0 / mean_time,
                'num_runs': num_runs
            }
            
            logger.info(f"Benchmark results: Mean={mean_time*1000:.2f}ms, "
                       f"Std={std_time*1000:.2f}ms, Throughput={1.0/mean_time:.2f} FPS")
            
            return benchmark_results
            
        except Exception as e:
            logger.error(f"Benchmarking failed: {e}")
            return {'error': str(e)}


def main():
    """CLI interface for ONNX export"""
    import argparse
    import yaml
    from ..models.efficientnet import create_efficientnet_b0
    from ..data.dataset_loader import load_class_mapping
    
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--opset-version", type=int, default=11, help="ONNX opset version")
    parser.add_argument("--optimize", action="store_true", help="Optimize the model")
    parser.add_argument("--quantize", type=str, choices=["int8", "uint8"], help="Quantize the model")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark the model")
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
    exporter = ONNXExporter(model, device)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export model
    onnx_path = output_dir / "model.onnx"
    exporter.export_model(onnx_path, opset_version=args.opset_version)
    
    # Optimize if requested
    if args.optimize:
        optimized_path = output_dir / "model_optimized.onnx"
        exporter.optimize_model(onnx_path, optimized_path)
        onnx_path = optimized_path
    
    # Quantize if requested
    if args.quantize:
        quantized_path = output_dir / f"model_quantized_{args.quantize}.onnx"
        exporter.quantize_model(onnx_path, quantized_path, args.quantize)
        onnx_path = quantized_path
    
    # Get model info
    model_info = exporter.get_model_info(onnx_path)
    
    # Validate model
    test_input = torch.randn(1, 3, 224, 224)
    validation_results = exporter.validate_model(onnx_path, test_input)
    
    # Benchmark if requested
    benchmark_results = None
    if args.benchmark:
        benchmark_results = exporter.benchmark_model(onnx_path)
    
    # Save results
    results = {
        'model_info': model_info,
        'validation_results': validation_results,
        'benchmark_results': benchmark_results,
        'class_names': class_names
    }
    
    with open(output_dir / "export_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"ONNX model exported to {onnx_path}")
    print(f"Model size: {model_info['model_size_mb']:.2f} MB")
    print(f"Validation: Model valid = {validation_results['model_valid']}, "
          f"Outputs close = {validation_results['outputs_close']}")
    
    if benchmark_results and 'error' not in benchmark_results:
        print(f"Benchmark: Mean inference time = {benchmark_results['mean_inference_time_ms']:.2f}ms, "
              f"Throughput = {benchmark_results['throughput_fps']:.2f} FPS")


if __name__ == "__main__":
    main()
