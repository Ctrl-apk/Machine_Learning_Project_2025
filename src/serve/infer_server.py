"""
FastAPI inference server for Sapling ML
Production-ready API for plant disease classification - because we need to serve this bitch
"""

import os
import io
import json
import base64
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging
import asyncio
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import uvicorn
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import yaml

# Import model classes
from ..models.efficientnet import create_efficientnet_b0
from ..models.mobilenetv3 import create_mobilenetv3_large
from ..models.resnet import create_resnet101
from ..explainability.gradcam import ExplainabilityAnalyzer

logger = logging.getLogger(__name__)


class PredictionRequest(BaseModel):
    """Request model for prediction - the sexy request format"""
    image_data: str = Field(..., description="Base64 encoded image data")
    include_explanation: bool = Field(False, description="Include Grad-CAM explanation")
    top_k: int = Field(5, description="Number of top predictions to return")


class PredictionResponse(BaseModel):
    """Response model for prediction - the fucking response format"""
    predictions: List[Dict[str, Any]] = Field(..., description="Top-k predictions")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    model_info: Dict[str, Any] = Field(..., description="Model information")
    explanation: Optional[Dict[str, Any]] = Field(None, description="Grad-CAM explanation if requested")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Current timestamp")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field(..., description="API version")


class ModelInfo(BaseModel):
    """Model information response"""
    architecture: str = Field(..., description="Model architecture")
    num_classes: int = Field(..., description="Number of classes")
    class_names: List[str] = Field(..., description="List of class names")
    input_shape: List[int] = Field(..., description="Input tensor shape")
    model_size_mb: float = Field(..., description="Model size in MB")


class InferenceServer:
    """Main inference server class - the sexy API server"""
    
    def __init__(self, config_path: str, model_path: str, device: str = "auto"):
        """
        Initialize inference server - because we need to serve this bitch somehow
        
        Args:
            config_path: Path to configuration file
            model_path: Path to trained model
            device: Device to run inference on
        """
        self.config_path = config_path
        self.model_path = model_path
        self.device = device
        self.model = None
        self.class_names = []
        self.class_mapping = {}
        self.explainability_analyzer = None
        self.model_info = {}
        
        # Load configuration
        self._load_config()
        
        # Setup device
        self._setup_device()
        
        # Load model
        self._load_model()
        
        # Create FastAPI app
        self.app = self._create_app()
    
    def _load_config(self):
        """Load configuration from file"""
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load class mapping
        class_names_config = self.config.get('class_names', {})
        self.class_mapping = {v: int(k) for k, v in class_names_config.items()}
        self.class_names = [name for name, _ in sorted(self.class_mapping.items(), key=lambda x: x[1])]
        
        logger.info(f"Loaded configuration with {len(self.class_names)} classes")
    
    def _setup_device(self):
        """Setup computation device"""
        if self.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.device)
        
        logger.info(f"Using device: {self.device}")
    
    def _load_model(self):
        """Load trained model"""
        logger.info("Loading model...")
        
        # Get model architecture from config
        model_config = self.config.get('model', {})
        architecture = model_config.get('architecture', 'efficientnet_b0')
        num_classes = model_config.get('num_classes', len(self.class_names))
        
        # Create model
        if architecture == 'efficientnet_b0':
            self.model = create_efficientnet_b0(num_classes=num_classes, pretrained=False)
        elif architecture == 'mobilenetv3_large':
            self.model = create_mobilenetv3_large(num_classes=num_classes, pretrained=False)
        elif architecture == 'resnet101':
            self.model = create_resnet101(num_classes=num_classes, pretrained=False)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        
        # Load weights
        checkpoint = torch.load(self.model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Create explainability analyzer
        target_layers = self.config.get('explainability', {}).get('gradcam', {}).get('target_layers', ['last_conv'])
        self.explainability_analyzer = ExplainabilityAnalyzer(self.model, self.class_names, target_layers)
        
        # Get model info
        self.model_info = {
            'architecture': architecture,
            'num_classes': num_classes,
            'class_names': self.class_names,
            'input_shape': [1, 3, 224, 224],
            'model_size_mb': Path(self.model_path).stat().st_size / (1024 * 1024)
        }
        
        logger.info(f"Model loaded successfully: {architecture} with {num_classes} classes")
    
    def _create_app(self) -> FastAPI:
        """Create FastAPI application"""
        app = FastAPI(
            title="Sapling ML API",
            description="Plant Disease Detection and Classification API",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Add routes
        self._add_routes(app)
        
        return app
    
    def _add_routes(self, app: FastAPI):
        """Add API routes"""
        
        @app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint"""
            return HealthResponse(
                status="healthy",
                timestamp=datetime.now().isoformat(),
                model_loaded=self.model is not None,
                version="1.0.0"
            )
        
        @app.get("/model/info", response_model=ModelInfo)
        async def get_model_info():
            """Get model information"""
            return ModelInfo(**self.model_info)
        
        @app.post("/predict", response_model=PredictionResponse)
        async def predict(request: PredictionRequest):
            """Predict plant disease from image"""
            try:
                start_time = datetime.now()
                
                # Decode image
                image_data = base64.b64decode(request.image_data)
                image = Image.open(io.BytesIO(image_data))
                
                # Preprocess image
                input_tensor = self._preprocess_image(image)
                
                # Run inference
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    probabilities = F.softmax(outputs, dim=1)
                
                # Get top-k predictions
                top_k = min(request.top_k, len(self.class_names))
                top_probs, top_indices = torch.topk(probabilities, top_k)
                
                predictions = []
                for i in range(top_k):
                    predictions.append({
                        'class_id': int(top_indices[0, i]),
                        'class_name': self.class_names[top_indices[0, i]],
                        'confidence': float(top_probs[0, i])
                    })
                
                # Generate explanation if requested
                explanation = None
                if request.include_explanation:
                    explanation = self._generate_explanation(input_tensor, int(top_indices[0, 0]))
                
                # Calculate processing time
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                
                return PredictionResponse(
                    predictions=predictions,
                    processing_time_ms=processing_time,
                    model_info=self.model_info,
                    explanation=explanation
                )
                
            except Exception as e:
                logger.error(f"Prediction failed: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
        
        @app.post("/predict/file", response_model=PredictionResponse)
        async def predict_file(file: UploadFile = File(...), 
                              include_explanation: bool = Form(False),
                              top_k: int = Form(5)):
            """Predict plant disease from uploaded file"""
            try:
                start_time = datetime.now()
                
                # Read image
                image_data = await file.read()
                image = Image.open(io.BytesIO(image_data))
                
                # Preprocess image
                input_tensor = self._preprocess_image(image)
                
                # Run inference
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    probabilities = F.softmax(outputs, dim=1)
                
                # Get top-k predictions
                top_k = min(top_k, len(self.class_names))
                top_probs, top_indices = torch.topk(probabilities, top_k)
                
                predictions = []
                for i in range(top_k):
                    predictions.append({
                        'class_id': int(top_indices[0, i]),
                        'class_name': self.class_names[top_indices[0, i]],
                        'confidence': float(top_probs[0, i])
                    })
                
                # Generate explanation if requested
                explanation = None
                if include_explanation:
                    explanation = self._generate_explanation(input_tensor, int(top_indices[0, 0]))
                
                # Calculate processing time
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                
                return PredictionResponse(
                    predictions=predictions,
                    processing_time_ms=processing_time,
                    model_info=self.model_info,
                    explanation=explanation
                )
                
            except Exception as e:
                logger.error(f"File prediction failed: {str(e)}")
                raise HTTPException(status_code=500, detail=f"File prediction failed: {str(e)}")
        
        @app.get("/classes")
        async def get_classes():
            """Get list of all classes"""
            return {
                'classes': [
                    {'id': i, 'name': name} 
                    for i, name in enumerate(self.class_names)
                ]
            }
        
        @app.get("/recommendations/{class_id}")
        async def get_recommendations(class_id: int):
            """Get treatment recommendations for a class"""
            if class_id < 0 or class_id >= len(self.class_names):
                raise HTTPException(status_code=404, detail="Class not found")
            
            class_name = self.class_names[class_id]
            recommendations = self._get_recommendations(class_name)
            
            return {
                'class_id': class_id,
                'class_name': class_name,
                'recommendations': recommendations
            }
    
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for model inference"""
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size
        image = image.resize((224, 224))
        
        # Convert to tensor and normalize
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        # Normalize with ImageNet mean and std
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_array = (image_array - mean) / std
        
        # Convert to tensor and add batch dimension
        input_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)
        
        return input_tensor
    
    def _generate_explanation(self, input_tensor: torch.Tensor, predicted_class: int) -> Dict[str, Any]:
        """Generate Grad-CAM explanation"""
        try:
            analysis = self.explainability_analyzer.analyze_prediction(input_tensor, predicted_class)
            
            # Extract relevant information
            explanation = {
                'predicted_class': analysis['predicted_class'],
                'predicted_class_name': analysis['predicted_class_name'],
                'confidence': analysis['confidence'],
                'gradcam': analysis['gradcam'],
                'gradcam_plus': analysis['gradcam_plus']
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Explanation generation failed: {str(e)}")
            return None
    
    def _get_recommendations(self, class_name: str) -> Dict[str, Any]:
        """Get treatment recommendations for a class"""
        recommendations_config = self.config.get('recommendations', {})
        
        # Basic recommendations
        cultural_practices = recommendations_config.get('cultural_practices', [])
        chemical_treatment = recommendations_config.get('chemical_treatment', {})
        monitoring = recommendations_config.get('monitoring', [])
        
        # Class-specific recommendations (can be extended)
        class_specific = {}
        
        # Check if it's a healthy class
        if 'healthy' in class_name.lower():
            class_specific = {
                'status': 'healthy',
                'message': 'Plant appears to be healthy. Continue current care practices.',
                'actions': [
                    'Maintain regular watering schedule',
                    'Continue monitoring for early signs of disease',
                    'Ensure proper nutrition and soil conditions'
                ]
            }
        else:
            # Disease-specific recommendations
            class_specific = {
                'status': 'diseased',
                'message': f'Plant shows signs of {class_name.replace("_", " ").lower()}.',
                'immediate_actions': cultural_practices[:3],  # First 3 cultural practices
                'long_term_actions': cultural_practices[3:],  # Remaining practices
                'monitoring': monitoring,
                'chemical_treatment': chemical_treatment
            }
        
        return {
            'general': {
                'cultural_practices': cultural_practices,
                'monitoring': monitoring,
                'chemical_treatment': chemical_treatment
            },
            'class_specific': class_specific
        }
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, workers: int = 1):
        """Run the inference server"""
        logger.info(f"Starting inference server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port, workers=workers)


def main():
    """CLI interface for inference server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Sapling ML inference server")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    
    args = parser.parse_args()
    
    # Create and run server
    server = InferenceServer(args.config, args.model_path, args.device)
    server.run(args.host, args.port, args.workers)


if __name__ == "__main__":
    main()
