import torch
from helpers.terminalColor import terminalColor as tc

    
def computePaTR(precision: torch.Tensor, recall: torch.Tensor) -> float:
    """Computes the precision at total recall (PaTR) for a given precision recall curve

    Args:
        precision (torch.Tensor): Precision values as Tensor
        recall (torch.Tensor): Recall values as Tensor

    Returns:
        float: PaTR value
    """
    idxRecallOne = torch.where(recall >= 1.0)[0]
    PaTR = float(precision[idxRecallOne[-1]])
    return PaTR


if __name__ == "__main__":
    print(tc.success + "finished!")