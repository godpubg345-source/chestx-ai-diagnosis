"""Model Evaluation and Metrics"""

import argparse
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from tqdm import tqdm

from ml.dataset import ChestXrayDataset
from ml.train import get_transforms
from app.models.densenet import ChestXrayDenseNet


def evaluate_model(model: nn.Module, test_loader: DataLoader, device: str = "cuda",
                   threshold: float = 0.5) -> Dict:
    model.eval()
    model.to(device)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.numpy())
    
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    
    diseases = ChestXrayDataset.DISEASES
    results = {'per_class': {}, 'overall': {}}
    
    for i, disease in enumerate(diseases):
        try:
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
        except ValueError:
            auc = 0.0
        
        try:
            ap = average_precision_score(y_true[:, i], y_pred[:, i])
        except ValueError:
            ap = 0.0
        
        results['per_class'][disease] = {'auc_roc': auc, 'avg_precision': ap}
    
    try:
        results['overall']['mean_auc'] = roc_auc_score(y_true, y_pred, average='macro')
    except ValueError:
        results['overall']['mean_auc'] = 0.0
    
    try:
        results['overall']['weighted_auc'] = roc_auc_score(y_true, y_pred, average='weighted')
    except ValueError:
        results['overall']['weighted_auc'] = 0.0
    
    return results, y_true, y_pred


def plot_roc_curves(y_true: np.ndarray, y_pred: np.ndarray,
                    save_path: str = "assets/roc_curves.png") -> None:
    diseases = ChestXrayDataset.DISEASES
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.flatten()
    
    for i, disease in enumerate(diseases):
        ax = axes[i]
        try:
            fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
            ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {auc:.3f}')
            ax.plot([0, 1], [0, 1], 'r--', linewidth=1)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(disease.replace('_', ' '))
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
        except ValueError:
            ax.set_title(f"{disease} (N/A)")
    
    if len(diseases) < len(axes):
        for i in range(len(diseases), len(axes)):
            fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curves to {save_path}")


def plot_auc_comparison(results: Dict, save_path: str = "assets/auc_comparison.png") -> None:
    diseases = list(results['per_class'].keys())
    aucs = [results['per_class'][d]['auc_roc'] for d in diseases]
    
    sorted_idx = np.argsort(aucs)[::-1]
    diseases = [diseases[i] for i in sorted_idx]
    aucs = [aucs[i] for i in sorted_idx]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ['#10b981' if auc >= 0.8 else '#f59e0b' if auc >= 0.6 else '#ef4444' for auc in aucs]
    bars = ax.barh(diseases, aucs, color=colors, edgecolor='white', linewidth=0.5)
    
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{auc:.3f}', va='center', fontsize=10)
    
    ax.set_xlabel('AUC-ROC Score', fontsize=12)
    ax.set_title('Model Performance by Disease', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.1)
    ax.axvline(x=0.8, color='green', linestyle='--', alpha=0.5, label='Good (>=0.8)')
    ax.axvline(x=0.6, color='orange', linestyle='--', alpha=0.5, label='Fair (>=0.6)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved AUC comparison to {save_path}")


def print_results(results: Dict) -> None:
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    
    print("\nPer-Class Metrics:")
    print("-"*40)
    
    for disease, metrics in results['per_class'].items():
        auc_val = metrics['auc_roc']
        ap_val = metrics['avg_precision']
        print(f"  {disease:20s} | AUC: {auc_val:.4f} | AP: {ap_val:.4f}")
    
    print("\nOverall Metrics:")
    print("-"*40)
    print(f"  Mean AUC:     {results['overall']['mean_auc']:.4f}")
    print(f"  Weighted AUC: {results['overall']['weighted_auc']:.4f}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate ChestX-AI Model")
    parser.add_argument('--model-path', type=str, default='models/best.pth')
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save-plots', action='store_true')
    args = parser.parse_args()
    
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    
    print("ChestX-AI Model Evaluation")
    print("="*60)
    
    print(f"Loading model from {args.model_path}...")
    model = ChestXrayDenseNet.load_pretrained(args.model_path, device)
    
    test_dataset = ChestXrayDataset(
        data_dir=args.data_dir, split='test',
        transform=get_transforms(is_training=False)
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    results, y_true, y_pred = evaluate_model(model, test_loader, device)
    print_results(results)
    
    if args.save_plots:
        Path("assets").mkdir(exist_ok=True)
        plot_roc_curves(y_true, y_pred)
        plot_auc_comparison(results)


if __name__ == "__main__":
    main()
