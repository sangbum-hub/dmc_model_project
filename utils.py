import torch 
import typing 

def proximity_correct(pred_y:torch.tensor, y_ture: torch.tensor) -> torch.tensor:
    '''
    Parameters
    ----------
    y_pred: predict label 
    y_ture: ground truth 

    Return 
    ----------
    acuuracy 
    '''
    p = torch.argmax(pred_y, dim=1)

    score = 0 
    for u, v in zip(p, y_ture):
        if v in [u-1, u, u+1]:
            score += 1 
    return score 

def return_values(x, score=True):
    category = ['A', 'B', 'C']
    idx = torch.argmax(x).cpu().item()
    if score:
        return 100 - 20*idx

    else:
        return category[idx]
