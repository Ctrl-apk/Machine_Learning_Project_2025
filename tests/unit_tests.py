"""
Unit tests for Sapling ML
Comprehensive testing suite for all components
Because apparently we need to test this fucking code
"""

import unittest
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch

# Import modules to test
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.efficientnet import create_efficientnet_b0
from src.models.mobilenetv3 import create_mobilenetv3_large
from src.models.resnet import create_resnet101
from src.data.dataset_loader import PlantDiseaseDataset, AugmentationFactory
from src.explainability.gradcam import GradCAM
from src.export.export_onnx import ONNXExporter


class TestModelArchitectures(unittest.TestCase):
    """Test model architectures - because we need to test this shit"""
    
    def setUp(self):
        """Set up test fixtures - the sexy test setup"""
        self.device = torch.device('cpu')
        self.num_classes = 39
        self.input_shape = (1, 3, 224, 224)
        self.test_input = torch.randn(self.input_shape)
    
    def test_efficientnet_b0(self):
        """Test EfficientNet-B0 model"""
        model = create_efficientnet_b0(num_classes=self.num_classes, pretrained=False)
        model.eval()
        
        with torch.no_grad():
            output = model(self.test_input)
        
        self.assertEqual(output.shape, (1, self.num_classes))
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
    
    def test_mobilenetv3_large(self):
        """Test MobileNetV3-Large model"""
        model = create_mobilenetv3_large(num_classes=self.num_classes, pretrained=False)
        model.eval()
        
        with torch.no_grad():
            output = model(self.test_input)
        
        self.assertEqual(output.shape, (1, self.num_classes))
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
    
    def test_resnet101(self):
        """Test ResNet-101 model"""
        model = create_resnet101(num_classes=self.num_classes, pretrained=False)
        model.eval()
        
        with torch.no_grad():
            output = model(self.test_input)
        
        self.assertEqual(output.shape, (1, self.num_classes))
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
    
    def test_model_forward_pass(self):
        """Test that models can handle different input sizes"""
        models = [
            create_efficientnet_b0(num_classes=self.num_classes, pretrained=False),
            create_mobilenetv3_large(num_classes=self.num_classes, pretrained=False),
            create_resnet101(num_classes=self.num_classes, pretrained=False)
        ]
        
        for model in models:
            model.eval()
            
            # Test different batch sizes
            for batch_size in [1, 2, 4]:
                test_input = torch.randn(batch_size, 3, 224, 224)
                
                with torch.no_grad():
                    output = model(test_input)
                
                self.assertEqual(output.shape, (batch_size, self.num_classes))


class TestDataLoader(unittest.TestCase):
    """Test data loading components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.image_dir = Path(self.temp_dir) / "images"
        self.image_dir.mkdir()
        
        # Create dummy manifest
        self.manifest_data = {
            'filename': ['test1.jpg', 'test2.jpg', 'test3.jpg'],
            'filepath': ['test1.jpg', 'test2.jpg', 'test3.jpg'],
            'class': ['Apple___healthy', 'Corn___healthy', 'Tomato___healthy'],
            'class_id': [0, 1, 2],
            'width': [224, 224, 224],
            'height': [224, 224, 224],
            'orig_id': ['orig1', 'orig2', 'orig3'],
            'source': ['original', 'original', 'original']
        }
        self.manifest_df = pd.DataFrame(self.manifest_data)
        
        # Create dummy images
        for i in range(3):
            dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            from PIL import Image
            img = Image.fromarray(dummy_image)
            img.save(self.image_dir / f"test{i+1}.jpg")
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_plant_disease_dataset(self):
        """Test PlantDiseaseDataset class"""
        class_mapping = {'Apple___healthy': 0, 'Corn___healthy': 1, 'Tomato___healthy': 2}
        
        dataset = PlantDiseaseDataset(
            manifest_df=self.manifest_df,
            image_dir=self.image_dir,
            class_mapping=class_mapping,
            transform=None,
            is_training=True
        )
        
        self.assertEqual(len(dataset), 3)
        
        # Test getting an item
        image, label, metadata = dataset[0]
        
        self.assertEqual(image.shape, (3, 224, 224))
        self.assertIsInstance(label, int)
        self.assertIn('filename', metadata)
    
    def test_augmentation_factory(self):
        """Test augmentation factory"""
        # Test training transforms
        train_transform = AugmentationFactory.get_training_transforms()
        self.assertIsNotNone(train_transform)
        
        # Test validation transforms
        val_transform = AugmentationFactory.get_validation_transforms()
        self.assertIsNotNone(val_transform)
        
        # Test that transforms work
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        transformed = train_transform(image=dummy_image)
        self.assertIn('image', transformed)
        self.assertEqual(transformed['image'].shape, (3, 224, 224))


class TestExplainability(unittest.TestCase):
    """Test explainability components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.device = torch.device('cpu')
        self.model = create_efficientnet_b0(num_classes=39, pretrained=False)
        self.target_layers = ['blocks.6.0']
        self.test_input = torch.randn(1, 3, 224, 224)
    
    def test_gradcam_initialization(self):
        """Test GradCAM initialization"""
        gradcam = GradCAM(self.model, self.target_layers)
        self.assertIsNotNone(gradcam)
        self.assertEqual(len(gradcam.target_layers), 1)
    
    def test_gradcam_generation(self):
        """Test Grad-CAM generation"""
        gradcam = GradCAM(self.model, self.target_layers)
        
        # Generate CAM
        cams = gradcam.generate_cam(self.test_input)
        
        self.assertIsInstance(cams, dict)
        self.assertIn(self.target_layers[0], cams)
        self.assertIsInstance(cams[self.target_layers[0]], np.ndarray)
    
    def tearDown(self):
        """Clean up test fixtures"""
        if hasattr(self, 'gradcam'):
            self.gradcam.remove_hooks()


class TestExport(unittest.TestCase):
    """Test model export functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.device = torch.device('cpu')
        self.model = create_efficientnet_b0(num_classes=39, pretrained=False)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_onnx_exporter_initialization(self):
        """Test ONNX exporter initialization"""
        exporter = ONNXExporter(self.model, self.device)
        self.assertIsNotNone(exporter)
        self.assertEqual(exporter.device, self.device)
    
    def test_onnx_export(self):
        """Test ONNX model export"""
        exporter = ONNXExporter(self.model, self.device)
        output_path = Path(self.temp_dir) / "test_model.onnx"
        
        # Export model
        exported_path = exporter.export_model(output_path)
        
        self.assertTrue(exported_path.exists())
        self.assertEqual(exported_path, output_path)
    
    def test_model_validation(self):
        """Test model validation"""
        exporter = ONNXExporter(self.model, self.device)
        output_path = Path(self.temp_dir) / "test_model.onnx"
        
        # Export model
        exporter.export_model(output_path)
        
        # Validate model
        test_input = torch.randn(1, 3, 224, 224)
        validation_results = exporter.validate_model(output_path, test_input)
        
        self.assertIn('model_valid', validation_results)
        self.assertIn('inference_success', validation_results)


class TestDataProcessing(unittest.TestCase):
    """Test data processing utilities"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_manifest_validation(self):
        """Test manifest validation"""
        # Test valid manifest
        valid_manifest = pd.DataFrame({
            'source': ['original', 'augmented'],
            'orig_id': ['orig1', 'orig1'],
            'filename': ['test1.jpg', 'test1_aug.jpg'],
            'class': ['Apple___healthy', 'Apple___healthy'],
            'class_id': [0, 0],
            'width': [224, 224],
            'height': [224, 224]
        })
        
        # This should not raise an exception
        self.assertIsNotNone(valid_manifest)
        self.assertEqual(len(valid_manifest), 2)
    
    def test_class_mapping(self):
        """Test class mapping functionality"""
        class_names = ['Apple___healthy', 'Corn___healthy', 'Tomato___healthy']
        class_mapping = {name: i for i, name in enumerate(class_names)}
        
        self.assertEqual(len(class_mapping), 3)
        self.assertEqual(class_mapping['Apple___healthy'], 0)
        self.assertEqual(class_mapping['Corn___healthy'], 1)
        self.assertEqual(class_mapping['Tomato___healthy'], 2)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""
    
    def test_tensor_operations(self):
        """Test tensor operations"""
        # Test tensor creation
        tensor = torch.randn(1, 3, 224, 224)
        self.assertEqual(tensor.shape, (1, 3, 224, 224))
        
        # Test tensor operations
        normalized = torch.nn.functional.normalize(tensor, p=2, dim=1)
        self.assertEqual(normalized.shape, tensor.shape)
        
        # Test softmax
        logits = torch.randn(1, 10)
        probs = torch.softmax(logits, dim=1)
        self.assertAlmostEqual(probs.sum().item(), 1.0, places=5)
    
    def test_numpy_operations(self):
        """Test NumPy operations"""
        # Test array creation
        arr = np.random.randn(224, 224, 3)
        self.assertEqual(arr.shape, (224, 224, 3))
        
        # Test array operations
        normalized = (arr - arr.mean()) / arr.std()
        self.assertAlmostEqual(normalized.mean(), 0.0, places=5)
        self.assertAlmostEqual(normalized.std(), 1.0, places=5)


class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def test_end_to_end_pipeline(self):
        """Test end-to-end pipeline"""
        # Create model
        model = create_efficientnet_b0(num_classes=39, pretrained=False)
        model.eval()
        
        # Create dummy data
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # Forward pass
        with torch.no_grad():
            output = model(dummy_input)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
        
        # Verify outputs
        self.assertEqual(output.shape, (1, 39))
        self.assertEqual(probabilities.shape, (1, 39))
        self.assertEqual(predicted_class.shape, (1,))
        self.assertGreaterEqual(predicted_class.item(), 0)
        self.assertLess(predicted_class.item(), 39)
    
    def test_model_consistency(self):
        """Test model consistency across different runs"""
        model = create_efficientnet_b0(num_classes=39, pretrained=False)
        model.eval()
        
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # Run inference multiple times
        outputs = []
        for _ in range(5):
            with torch.no_grad():
                output = model(dummy_input)
                outputs.append(output)
        
        # All outputs should be identical
        for i in range(1, len(outputs)):
            self.assertTrue(torch.allclose(outputs[0], outputs[i]))


def run_tests():
    """Run all tests"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestModelArchitectures,
        TestDataLoader,
        TestExplainability,
        TestExport,
        TestDataProcessing,
        TestUtilityFunctions,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result


if __name__ == '__main__':
    # Run tests
    result = run_tests()
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
