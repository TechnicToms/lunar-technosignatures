import lightning as pl

import torch
from attribution.gradcam import grad_cam
import torchvision
from lightning.pytorch.loggers import TensorBoardLogger

from datasets.DatasetFromImage import DatasetFromImage

import tqdm
import os
import json
from helpers.terminalColor import terminalColor as tc

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from sklearn.metrics import average_precision_score
from metrics.computePaTR import computePaTR


def evaluateLandingSite(model: pl.LightningModule, dataset: DatasetFromImage, data_root: str):
    model.eval()
    
    resizeLayer = torchvision.transforms.Resize(dataset.imgSize)
    
    scoreMap = torch.zeros(dataset.num_cols, dataset.num_rows)
    scoreMapTruth = torch.zeros(dataset.num_cols, dataset.num_rows)
    globalAnoMap = torch.zeros(dataset.img.shape[1], dataset.img.shape[2])
    ground_truth = torchvision.io.read_image(os.path.join(data_root, "gt_descent_stage.png"))[0, ...] / 255.0
    
    y_pred = torch.zeros(len(dataset))
    y_true = torch.zeros(len(dataset))
    y_score = torch.zeros(len(dataset))
    
    for it, img in tqdm.tqdm(enumerate(dataset), desc="Evaluating", total=len(dataset)):
        batch = torch.stack([img]*3, dim=1).to(model.device)
        cx, cy = dataset.patchIds[:, it]
        current_gt = ground_truth[cy:cy+dataset.imgSize, cx:cx+dataset.imgSize]
        
        model.zero_grad()
        anomalyMap, prediction = grad_cam(model=model, input_tensor=batch[0, ...], heatmap_layer=model.backbone.layer1[0].conv2, truelabel=0)
        
        y_true[it] = 1.0 if torch.sum(current_gt) > 100.0 else 0.0
        y_pred[it] = torch.argmax(prediction)
        y_score[it] = prediction[1]
        
        globalAnoMap[cy:cy+dataset.imgSize, cx:cx+dataset.imgSize] += resizeLayer(anomalyMap[None, None, ...])[0, 0, ...]
        scoreMap[it//dataset.num_rows, it%dataset.num_rows] = prediction[1].cpu()
        scoreMapTruth[it//dataset.num_rows, it%dataset.num_rows] = y_true[it]

    # Compute optimal threshold and ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true=y_true.detach().cpu(), y_score=y_score.detach().cpu())
    auc = roc_auc_score(y_true=y_true.detach().cpu(), y_score=y_score.detach().cpu())
    # optimal_idx = torch.argmax(torch.sqrt((1 - torch.from_numpy(fpr))**2 + torch.from_numpy(tpr)**2))
    # optimal_threshold = thresholds[optimal_idx]
    
    plt.plot(fpr, tpr)
    plt.title("ROC")
    plt.xlabel("fpr")
    plt.ylabel("tpr")
    plt.savefig(os.path.join("results", "ROC.png"))
    plt.close()

    precision_curve, recall_curve, _ = precision_recall_curve(y_true.detach().cpu(), y_score.detach().cpu())
    
    plt.plot(recall_curve, precision_curve)
    plt.title("ROC")
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.savefig(os.path.join("results", "PR.png"))
    plt.close()
    
    plt.imsave("results/scores.png", scoreMap.detach().numpy())
    plt.imsave("results/scores_gt.png", scoreMapTruth.detach().numpy())
    plt.imsave("results/ano.png", globalAnoMap.detach().numpy())
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    PaTR = computePaTR(precision=torch.from_numpy(precision_curve), recall=torch.from_numpy(recall_curve))
    AP = average_precision_score(y_true=y_true, y_score=y_pred)
        
    print(tc.info + f"Accuracy: {accuracy:.6f}")
    print(tc.info + f"Precision: {precision:.6f}")
    print(tc.info + f"Recall: {recall:.6f}")
    print(tc.info + f"AP: {AP:.6f}")

    metrics = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "PaTR": PaTR, "AUC": auc, "AP": AP}
    
    with open(os.path.join("results", 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    print(tc.success + "finished!")