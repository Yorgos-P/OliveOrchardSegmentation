from deepchecks.vision.checks import ImagePropertyDrift
from data.dataset_loader import OliveOrchard
import torch
from torch.utils.data import DataLoader
from deepchecks.vision import VisionData, BatchOutputFormat
from deepchecks.vision.suites import train_test_validation
from deepchecks.vision.checks import ImageDatasetDrift
import json, pathlib
import torch
from torch.utils.data import DataLoader
from deepchecks.vision import VisionData, BatchOutputFormat

def deepchecks_collate(data) -> BatchOutputFormat:
    # Extracting images and label and converting images of (N, C, H, W) into (N, H, W, C)
    images = torch.stack([x[0] for x in data]).permute(0, 2, 3, 1)
    # Ensure masks are HxW integer tensors (Deepchecks expects 2D per-sample labels)
    labels = []
    for _, mask in data:
        # mask is typically [1, H, W] from torchvision.read_image; squeeze channel dim only
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)
        # If batched by mistake, reduce batch dim
        if mask.ndim == 3 and mask.shape[-3] == 1:
            mask = mask[0]
        # Cast to integer type
        mask = mask.to(torch.long)
        labels.append(mask)
    return BatchOutputFormat(images= images, labels= labels)

def run_drift_report():
    data_loader_train = DataLoader(OliveOrchard("training"), batch_size=4, shuffle=False, num_workers=0, collate_fn=deepchecks_collate)
    data_loader_val = DataLoader(OliveOrchard("val"), batch_size=4, shuffle=False, num_workers=0, collate_fn=deepchecks_collate)

    vision_data_train = VisionData(data_loader_train, task_type='semantic_segmentation')
    vision_data_eval = VisionData(data_loader_val, task_type='semantic_segmentation')
    # Avoid joblib multiprocessing issues on macOS/Windows by running single-threaded
    check = ImageDatasetDrift(n_jobs=1)
    result = check.run(train_dataset=vision_data_train, test_dataset=vision_data_eval)
    # Prefer saving HTML to ensure it renders outside notebooks
    json_path = 'deepchecks_imagedrift_report.json'
    with open(json_path, 'w') as f:
        json.dump(result.to_json(), f, indent=2)
        print(f"Saved Deepchecks JSON to: {pathlib.Path(json_path).resolve()}")


if __name__ == "__main__":
    run_drift_report()
