"""
Evaluation utilities for Sapling ML
Comprehensive evaluation metrics and cross-dataset testing
Because apparently we need to know if this bitch actually works
"""

import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation class - the evaluation bitch"""
    
    def __init__(self, model: torch.nn.Module, device: torch.device, class_names: List[str]):
        """
        Initialize model evaluator - because we need to test this fucking model
        
        Args:
            model: Trained PyTorch model
            device: Device to run evaluation on
            class_names: List of class names
        """
        self.model = model
        self.device = device
        self.class_names = class_names
        self.num_classes = len(class_names)
        
        # Set model to evaluation mode
        self.model.eval()
    
    def evaluate_dataset(self, data_loader: torch.utils.data.DataLoader, 
                        dataset_name: str = "test") -> Dict[str, Any]:
        """
        Evaluate model on a dataset
        
        Args:
            data_loader: DataLoader for the dataset
            dataset_name: Name of the dataset for logging
            
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info(f"Evaluating on {dataset_name} dataset")
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        all_metadata = []
        
        with torch.no_grad():
            for images, labels, metadata in tqdm(data_loader, desc=f"Evaluating {dataset_name}"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_metadata.extend(metadata)
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_labels, all_predictions, all_probabilities)
        
        # Add dataset info
        metrics['dataset_name'] = dataset_name
        metrics['num_samples'] = len(all_labels)
        
        logger.info(f"Evaluation completed for {dataset_name}: "
                   f"Accuracy={metrics['accuracy']:.4f}, "
                   f"Macro F1={metrics['macro_f1']:.4f}")
        
        return metrics
    
    def _calculate_metrics(self, labels: np.ndarray, predictions: np.ndarray, 
                          probabilities: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics"""
        metrics = {}
        
        # Basic accuracy
        metrics['accuracy'] = accuracy_score(labels, predictions)
        
        # Precision, Recall, F1
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )
        
        # Macro averages
        metrics['macro_precision'] = np.mean(precision)
        metrics['macro_recall'] = np.mean(recall)
        metrics['macro_f1'] = np.mean(f1)
        
        # Weighted averages
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        metrics['weighted_precision'] = precision_weighted
        metrics['weighted_recall'] = recall_weighted
        metrics['weighted_f1'] = f1_weighted
        
        # Per-class metrics
        metrics['per_class_precision'] = precision.tolist()
        metrics['per_class_recall'] = recall.tolist()
        metrics['per_class_f1'] = f1.tolist()
        metrics['per_class_support'] = support.tolist()
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(labels, predictions).tolist()
        
        # ROC AUC (one-vs-rest)
        try:
            if self.num_classes > 2:
                roc_auc = roc_auc_score(labels, probabilities, multi_class='ovr', average='macro')
                metrics['macro_roc_auc'] = roc_auc
                
                # Per-class ROC AUC
                roc_auc_per_class = []
                for i in range(self.num_classes):
                    if i in labels:  # Only calculate if class exists in labels
                        class_labels = (labels == i).astype(int)
                        if len(np.unique(class_labels)) > 1:  # Check if class has both positive and negative samples
                            auc = roc_auc_score(class_labels, probabilities[:, i])
                            roc_auc_per_class.append(auc)
                        else:
                            roc_auc_per_class.append(0.0)
                    else:
                        roc_auc_per_class.append(0.0)
                metrics['per_class_roc_auc'] = roc_auc_per_class
            else:
                roc_auc = roc_auc_score(labels, probabilities[:, 1])
                metrics['roc_auc'] = roc_auc
        except Exception as e:
            logger.warning(f"Could not calculate ROC AUC: {str(e)}")
            metrics['macro_roc_auc'] = 0.0
        
        # Top-k accuracy
        for k in [2, 3, 5]:
            if k <= self.num_classes:
                top_k_acc = self._calculate_top_k_accuracy(labels, probabilities, k)
                metrics[f'top_{k}_accuracy'] = top_k_acc
        
        return metrics
    
    def _calculate_top_k_accuracy(self, labels: np.ndarray, probabilities: np.ndarray, k: int) -> float:
        """Calculate top-k accuracy"""
        top_k_predictions = np.argsort(probabilities, axis=1)[:, -k:]
        correct = 0
        for i, label in enumerate(labels):
            if label in top_k_predictions[i]:
                correct += 1
        return correct / len(labels)
    
    def generate_classification_report(self, labels: np.ndarray, predictions: np.ndarray) -> str:
        """Generate detailed classification report"""
        return classification_report(
            labels, predictions, 
            target_names=self.class_names,
            digits=4
        )
    
    def plot_confusion_matrix(self, labels: np.ndarray, predictions: np.ndarray, 
                            save_path: Optional[Path] = None, figsize: Tuple[int, int] = (12, 10)):
        """Plot confusion matrix"""
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_per_class_metrics(self, metrics: Dict[str, Any], 
                              save_path: Optional[Path] = None, figsize: Tuple[int, int] = (15, 5)):
        """Plot per-class precision, recall, and F1 scores"""
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Precision
        axes[0].bar(range(len(self.class_names)), metrics['per_class_precision'])
        axes[0].set_title('Per-Class Precision')
        axes[0].set_xlabel('Class')
        axes[0].set_ylabel('Precision')
        axes[0].set_xticks(range(len(self.class_names)))
        axes[0].set_xticklabels(self.class_names, rotation=45, ha='right')
        
        # Recall
        axes[1].bar(range(len(self.class_names)), metrics['per_class_recall'])
        axes[1].set_title('Per-Class Recall')
        axes[1].set_xlabel('Class')
        axes[1].set_ylabel('Recall')
        axes[1].set_xticks(range(len(self.class_names)))
        axes[1].set_xticklabels(self.class_names, rotation=45, ha='right')
        
        # F1 Score
        axes[2].bar(range(len(self.class_names)), metrics['per_class_f1'])
        axes[2].set_title('Per-Class F1 Score')
        axes[2].set_xlabel('Class')
        axes[2].set_ylabel('F1 Score')
        axes[2].set_xticks(range(len(self.class_names)))
        axes[2].set_xticklabels(self.class_names, rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Per-class metrics plot saved to {save_path}")
        
        plt.show()
    
    def evaluate_cross_dataset(self, data_loaders: Dict[str, torch.utils.data.DataLoader]) -> Dict[str, Any]:
        """Evaluate model on multiple datasets"""
        logger.info("Starting cross-dataset evaluation")
        
        results = {}
        
        for dataset_name, data_loader in data_loaders.items():
            logger.info(f"Evaluating on {dataset_name}")
            metrics = self.evaluate_dataset(data_loader, dataset_name)
            results[dataset_name] = metrics
        
        # Generate summary
        summary = self._generate_cross_dataset_summary(results)
        results['summary'] = summary
        
        return results
    
    def _generate_cross_dataset_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics across datasets"""
        summary = {
            'datasets': list(results.keys()),
            'accuracy_by_dataset': {},
            'macro_f1_by_dataset': {},
            'best_dataset': None,
            'worst_dataset': None
        }
        
        # Extract key metrics
        for dataset_name, metrics in results.items():
            summary['accuracy_by_dataset'][dataset_name] = metrics['accuracy']
            summary['macro_f1_by_dataset'][dataset_name] = metrics['macro_f1']
        
        # Find best and worst performing datasets
        if summary['accuracy_by_dataset']:
            best_acc = max(summary['accuracy_by_dataset'].values())
            worst_acc = min(summary['accuracy_by_dataset'].values())
            
            summary['best_dataset'] = max(summary['accuracy_by_dataset'], key=summary['accuracy_by_dataset'].get)
            summary['worst_dataset'] = min(summary['accuracy_by_dataset'], key=summary['accuracy_by_dataset'].get)
            
            summary['accuracy_range'] = best_acc - worst_acc
            summary['mean_accuracy'] = np.mean(list(summary['accuracy_by_dataset'].values()))
            summary['std_accuracy'] = np.std(list(summary['accuracy_by_dataset'].values()))
        
        return summary
    
    def save_evaluation_results(self, results: Dict[str, Any], save_dir: Path):
        """Save evaluation results to files"""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results as JSON
        with open(save_dir / "evaluation_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save summary as CSV
        summary_data = []
        for dataset_name, metrics in results.items():
            if dataset_name != 'summary':
                summary_data.append({
                    'dataset': dataset_name,
                    'accuracy': metrics['accuracy'],
                    'macro_f1': metrics['macro_f1'],
                    'macro_precision': metrics['macro_precision'],
                    'macro_recall': metrics['macro_recall'],
                    'num_samples': metrics['num_samples']
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(save_dir / "evaluation_summary.csv", index=False)
        
        logger.info(f"Evaluation results saved to {save_dir}")


class CrossDatasetEvaluator:
    """Specialized evaluator for cross-dataset testing"""
    
    def __init__(self, model: torch.nn.Module, device: torch.device, class_names: List[str]):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.evaluator = ModelEvaluator(model, device, class_names)
    
    def evaluate_plantdoc(self, plantdoc_loader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """Evaluate on PlantDoc dataset (out-of-domain)"""
        logger.info("Evaluating on PlantDoc dataset (out-of-domain)")
        
        # Run evaluation
        metrics = self.evaluator.evaluate_dataset(plantdoc_loader, "plantdoc")
        
        # Add domain-specific analysis
        metrics['domain'] = 'out_of_domain'
        metrics['dataset_type'] = 'field_images'
        
        return metrics
    
    def evaluate_regional_datasets(self, regional_loaders: Dict[str, torch.utils.data.DataLoader]) -> Dict[str, Any]:
        """Evaluate on regional datasets"""
        logger.info("Evaluating on regional datasets")
        
        results = {}
        for region_name, loader in regional_loaders.items():
            metrics = self.evaluator.evaluate_dataset(loader, region_name)
            metrics['domain'] = 'regional'
            metrics['region'] = region_name
            results[region_name] = metrics
        
        return results
    
    def analyze_domain_gap(self, in_domain_results: Dict[str, Any], 
                          out_domain_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the gap between in-domain and out-of-domain performance"""
        logger.info("Analyzing domain gap")
        
        # Extract key metrics
        in_domain_acc = np.mean([metrics['accuracy'] for metrics in in_domain_results.values()])
        out_domain_acc = np.mean([metrics['accuracy'] for metrics in out_domain_results.values()])
        
        in_domain_f1 = np.mean([metrics['macro_f1'] for metrics in in_domain_results.values()])
        out_domain_f1 = np.mean([metrics['macro_f1'] for metrics in out_domain_results.values()])
        
        gap_analysis = {
            'in_domain_accuracy': in_domain_acc,
            'out_domain_accuracy': out_domain_acc,
            'accuracy_gap': in_domain_acc - out_domain_acc,
            'in_domain_f1': in_domain_f1,
            'out_domain_f1': out_domain_f1,
            'f1_gap': in_domain_f1 - out_domain_f1,
            'domain_robustness': out_domain_acc / in_domain_acc if in_domain_acc > 0 else 0
        }
        
        logger.info(f"Domain gap analysis: Accuracy gap={gap_analysis['accuracy_gap']:.4f}, "
                   f"F1 gap={gap_analysis['f1_gap']:.4f}")
        
        return gap_analysis


def main():
    """CLI interface for model evaluation"""
    import argparse
    import yaml
    from ..models.efficientnet import create_efficientnet_b0
    from ..data.dataset_loader import DataLoaderFactory, load_class_mapping
    
    parser = argparse.ArgumentParser(description="Evaluate Sapling ML model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--test-split", type=str, required=True, help="Path to test split CSV")
    parser.add_argument("--image-dir", type=str, required=True, help="Base image directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for results")
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
    
    # Evaluate model
    evaluator = ModelEvaluator(model, device, class_names)
    results = evaluator.evaluate_dataset(test_loader, "test")
    
    # Save results
    output_dir = Path(args.output_dir)
    evaluator.save_evaluation_results({"test": results}, output_dir)
    
    print(f"Evaluation completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
