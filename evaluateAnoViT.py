import lightning as pl

import torch
import torchvision

from datasets.DatasetFromImage import DatasetFromImage

import os
import json
import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from sklearn.metrics import average_precision_score, f1_score
from metrics.computePaTR import computePaTR

from helpers.terminalColor import terminalColor as tc


def evaluateAnoViT(save_results_root: str, model: pl.LightningModule, dataset: DatasetFromImage, data_root: str):
    model.eval()
    
    resizeLayer = torchvision.transforms.Resize(dataset.imgSize)
    
    scoreMap = torch.zeros(dataset.num_cols, dataset.num_rows)
    scoreMapTruth = torch.zeros(dataset.num_cols, dataset.num_rows)
    globalAnoMap = torch.zeros(dataset.img.shape[1], dataset.img.shape[2])
    ground_truth = torchvision.io.read_image(os.path.join(data_root, "gt_descent_stage.png"))[0, ...] / 255.0
    
    y_true = torch.zeros(len(dataset))
    y_score = torch.zeros(len(dataset))
    
    for it, img in tqdm.tqdm(enumerate(dataset), desc="Evaluating", total=len(dataset)):
        batch = torch.stack([img]*3, dim=1).to(model.device)
        cx, cy = dataset.patchIds[:, it]
        current_gt = ground_truth[cy:cy+dataset.imgSize, cx:cx+dataset.imgSize]
        
        model.zero_grad()
        with torch.no_grad():
            reconstructed_image = model(batch)
        anomaly_map = torch.abs(reconstructed_image - batch)
        
        y_true[it] = 1.0 if torch.sum(current_gt) > 100.0 else 0.0
        y_score[it] = torch.max(anomaly_map)
        
        globalAnoMap[cy:cy+dataset.imgSize, cx:cx+dataset.imgSize] += resizeLayer(anomaly_map.cpu())[0, 0, ...]
        scoreMap[it//dataset.num_rows, it%dataset.num_rows] =torch.max(anomaly_map).cpu()
        scoreMapTruth[it//dataset.num_rows, it%dataset.num_rows] = y_true[it]

    # Compute optimal threshold and ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_score)
    auc = roc_auc_score(y_true=y_true, y_score=y_score)
    optimal_idx = torch.argmax(torch.sqrt((1 - torch.from_numpy(fpr))**2 + torch.from_numpy(tpr)**2))
    optimal_threshold = thresholds[optimal_idx]
    
    plt.plot(fpr, tpr)
    plt.title("ROC")
    plt.xlabel("fpr")
    plt.ylabel("tpr")
    plt.savefig(os.path.join(save_results_root, "ROC.png"))
    plt.close()
    
    # Compute precision recall curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_score)
    
    plt.plot(recall_curve, precision_curve)
    plt.title("ROC")
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.savefig(os.path.join(save_results_root, "PR.png"))
    plt.close()

    plt.imsave(os.path.join(save_results_root, "scores.png"), scoreMap.detach().numpy())
    plt.imsave(os.path.join(save_results_root, "scores_gt.png"), scoreMapTruth.detach().numpy())
    plt.imsave(os.path.join(save_results_root, "ano.png"), globalAnoMap.detach().numpy())
    
    y_pred = torch.where(y_score > optimal_threshold, 1.0, 0.0)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    PaTR = computePaTR(precision=torch.from_numpy(precision_curve), recall=torch.from_numpy(recall_curve))
    ap = average_precision_score(y_true=y_true, y_score=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred)
        
    print(tc.info + f"Accuracy: {accuracy:.6f}")
    print(tc.info + f"Precision: {precision:.6f}")
    print(tc.info + f"Recall: {recall:.6f}")
    print(tc.info + f"AP: {ap:.6f}")
    print(tc.info + f"F1 {f1:6f}")
    
    metrics = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "PaTR": PaTR, "AUC": auc, "AP": ap, "F1": f1}
    
    with open(os.path.join(save_results_root, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    from anoViT import AnoViT
    from datasets.Apollo15 import constructApollo15LandingSite
    
    hparams = {
        # Data related
        'root': '/home/tsa/data/LRO/NAC/Apollo 15/',
        'imgSize': 224,
        'stride': 28,
        
        # Transformer related
        'patch_size': 16, 
        'num_layers': 4,
        'num_heads': 4,
        'hidden_dim': 128,
        'embed_dim': 128,
        
        # Training related
        'lr': 1e-4,
        'batch_size': 64,
        'num_epochs': 75,
        
        # Logger
        'loggerTag': 'AnoViT',
    }

    model = AnoViT.load_from_checkpoint("tb_logs/AnoViT/version_1/checkpoints/AnoViT.ckpt", hparams=hparams)
    model.eval()
    
    test_data = constructApollo15LandingSite(root=hparams["root"], imgSize=hparams["imgSize"], stride=8)
    evaluateAnoViT("results", model=model, dataset=test_data, data_root=hparams["root"])
    
    print(tc.success + 'finished!')