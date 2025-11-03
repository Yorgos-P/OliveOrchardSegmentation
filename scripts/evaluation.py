from visualisation.val_overlays import forward_pass
from data.augmentations import val_transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.v2 as v2
from torchvision.io import decode_image
from torchvision.tv_tensors import Image as TVImage, Mask as TVMask
from data.dataset_loader import OliveOrchard
import json
import torch

def mask_processing(mask):
    #We want the target mask to match the pred mask
    mask = v2.Resize(size = (768, 768), interpolation = InterpolationMode.NEAREST)(mask)
    return mask


val_ds = OliveOrchard('val', transform = None)




def IoU(pred_mask_bool, target_mask_bool, cls):
    intersection = (pred_mask_bool & target_mask_bool).sum()
    union = (pred_mask_bool | target_mask_bool).sum()
    iou = intersection/union
    return iou.item()

def precision(pred_mask_bool, target_mask_bool, cls):
    tp = (pred_mask_bool & target_mask_bool).sum()
    tp_and_fp = pred_mask_bool.sum()
    precision = tp/tp_and_fp
    return precision.item()

def recall(pred_mask_bool, target_mask_bool, cls):
    tp = (pred_mask_bool & target_mask_bool).sum()
    fn = (~pred_mask_bool & target_mask_bool).sum()
    recall = tp/(tp+fn)
    return recall.item()


def eval(loader, cls):
    m_IoU = 0
    m_precision = 0
    m_recall = 0
    count = len(loader)

    for idx in range(len(loader)):
        img, mask = loader[idx]
        pred_mask = forward_pass(img, eval_mode = True)
        pred_mask_bool = pred_mask == cls
        target_mask_bool = mask_processing(mask) == cls
        if (target_mask_bool.sum() == 0) & (cls == 1): #This is to skip hard negatives because they give undefined recall when evaluating non-background classes
            count -=1
            continue
        m_IoU += IoU(pred_mask_bool, target_mask_bool, cls)
        m_precision += precision(pred_mask_bool, target_mask_bool, cls)
        m_recall += recall(pred_mask_bool, target_mask_bool, cls)
    print(count)
    return (m_IoU/count, m_precision/count, m_recall/count)

cls_0 = eval(val_ds, 0)
cls_1 = eval(val_ds, 1)

def save_results():
    results = {
        "mean_iou" : (cls_0[0] + cls_1[0])/2,
        "mean_precision": (cls_0[1] + cls_1[1])/2,
        "mean_recall": (cls_0[2] + cls_1[2])/2,
        "per_class":{
            0: {"iou_bg": cls_0[0], "precision_bg": cls_0[1], "recall_bg": cls_0[2]},
            1: {"iou_tree": cls_1[0], "precision_tree": cls_1[1], "recall_tree": cls_1[2]}
        }
    }
    with open("evaluation_results.json", "w") as fp:
        json.dump(results, fp, indent = 2)


save_results()





