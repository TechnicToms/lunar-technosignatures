import lightning as pl

import torch
from torch.utils.data import DataLoader
import torchvision

from datasets.DatasetFromImage import DatasetFromImage

import os
import json
import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from sklearn.metrics import average_precision_score
from metrics.computePaTR import computePaTR

from helpers.terminalColor import terminalColor as tc


def evaluateEffAD(save_results_root: str, model: pl.LightningModule, dataset: DatasetFromImage, data_root: str):
    trainer = pl.Trainer()
    loader = DataLoader(dataset, shuffle=False, batch_size=1)
    model.eval()
    
    resizeLayer = torchvision.transforms.Resize(dataset.imgSize)
    
    scoreMap = torch.zeros(dataset.num_cols, dataset.num_rows)
    scoreMapTruth = torch.zeros(dataset.num_cols, dataset.num_rows)
    globalAnoMap = torch.zeros(dataset.img.shape[1], dataset.img.shape[2])
    ground_truth = torchvision.io.read_image(os.path.join(data_root, "gt_descent_stage.png"))[0, ...] / 255.0
    
    y_true = torch.zeros(len(dataset))
    y_score = torch.zeros(len(dataset))
    
    model_dicts = trainer.predict(model, loader)
    pad_img = 30

    for i in range(len(model_dicts)):
        pred_anomalyMap: torch.Tensor = model_dicts[i]['anomaly_map']
        pred_anomalyMap_student: torch.Tensor  = model_dicts[i]['map_st']
        pred_anomalyMap_autoencoder: torch.Tensor  = model_dicts[i]['map_ae']

        cx, cy = dataset.patchIds[:, i]
        current_gt = ground_truth[cy:cy+dataset.imgSize, cx:cx+dataset.imgSize]
        
        y_true[i] = 1.0 if torch.sum(current_gt) > 100.0 else 0.0
        y_score[i] = torch.max(pred_anomalyMap)
        
        globalAnoMap[cy+pad_img//2:cy+dataset.imgSize-pad_img//2, cx+pad_img//2:cx+dataset.imgSize -pad_img//2] += pred_anomalyMap_student.cpu()[0, 0, pad_img//2:-pad_img//2, pad_img//2:-pad_img//2]
        scoreMap[i//dataset.num_rows, i%dataset.num_rows] =torch.max(pred_anomalyMap_student).cpu()
        scoreMapTruth[i//dataset.num_rows, i%dataset.num_rows] = y_true[i]

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
    AP = average_precision_score(y_true=y_true, y_score=y_pred)
        
    print(tc.info + f"Accuracy: {accuracy:.6f}")
    print(tc.info + f"Precision: {precision:.6f}")
    print(tc.info + f"Recall: {recall:.6f}")
    print(tc.info + f"AP: {AP:.6f}")
    
    metrics = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "PaTR": PaTR, "AUC": auc, "AP": AP}
    
    with open(os.path.join(save_results_root, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    from EfficientAD import EffAD, EfficientAdModelSize
    from datasets.Apollo15 import constructApollo15LandingSite
    
    hparams = {
        # Data related
        'root': '/home/tsa/data/LRO/NAC/Apollo 15/',
        'root_imageNet': '/home/tsa/data/imagenet/ILSVRC/Data/CLS-LOC/train',
        'imgSize': 256,
        'stride': 8,
        
        # Model related stuff
        'teacher_channels': 384,
        'model_size': EfficientAdModelSize.S,
        'padding': True,
        'teacher_ckpt': 'models/teacher_checkpoints/teacher_small_final_state.pth',
        
        # Training related
        'lr': 1e-4,
        'batch_size': 32,
        'num_epochs': 75,
        
        # Logger
        'loggerTag': 'EffAD',
    }

    model = EffAD.load_from_checkpoint("tb_logs/EffAD/version_0/checkpoints/effAD.ckpt", hparams=hparams)
    model.eval()
    
    transform_lunar = torchvision.transforms.Normalize(mean=[0.2987], std=[0.0922])
    test_data = constructApollo15LandingSite(root=hparams["root"], imgSize=hparams["imgSize"], stride=hparams['stride'], transform=transform_lunar)
    evaluateEffAD("results", model=model, dataset=test_data, data_root=hparams["root"])
    
    print(tc.success + 'finished!')