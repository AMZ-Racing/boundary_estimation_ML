import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np

def order_points_pca(points: np.ndarray):
    '''used chatgpt'''
    if len(points) < 2:
        return points
    pts = points - np.mean(points, axis=0)
    U, S, Vt = np.linalg.svd(pts, full_matrices=False)
    direction = Vt[0]
    proj = pts @ direction
    ordering = np.argsort(proj)
    return points[ordering]


def chamfer_distance(a: np.ndarray, b: np.ndarray):
    '''
    used chatgpt
    '''
    if len(a) == 0 and len(b) == 0:
        return 0.0
    if len(a) == 0 or len(b) == 0:
        return float('inf')
    diff1 = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2)  # (Na, Nb)
    diff2 = diff1.T
    return float(np.mean(np.min(diff1, axis=1)) + np.mean(np.min(diff2, axis=1)))


def hausdorff_distance(a: np.ndarray, b: np.ndarray):
    '''
    used chatgpt
    '''
    if len(a) == 0 and len(b) == 0:
        return 0.0
    if len(a) == 0 or len(b) == 0:
        return float('inf')
    diff = np.sqrt(np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2))
    d1 = np.max(np.min(diff, axis=1))
    d2 = np.max(np.min(diff, axis=0))
    return float(max(d1, d2))



def validate(model: nn.Module, val_loader: DataLoader, device: torch.device, id_left=0, id_right=1):
    model.eval()
    all_preds = []
    all_targets = []

    chamfers_left = []
    chamfers_right = []
    hausdorff_left = []
    hausdorff_right = []

    with torch.no_grad():
        for points, labels, mask, lengths in val_loader:
            points = points.to(device)        # (B, N, 2)
            labels = labels.to(device)        # (B, N)
            mask = mask.to(device)            # (B, N) bool (True=PAD)

            logits = model(points, mask)      # (B, N, C)
            preds = torch.argmax(logits, dim=-1)  # (B, N)

            preds_np = preds.cpu().numpy()
            labels_np = labels.cpu().numpy()
            mask_np = mask.cpu().numpy()

            for b in range(points.shape[0]):
                valid_idx = ~mask_np[b]
                all_preds.append(preds_np[b, valid_idx])
                all_targets.append(labels_np[b, valid_idx])

            pts_np_b = points.cpu().numpy()
            for b in range(points.shape[0]):
                pts_np = pts_np_b[b]  # (N,2)
                pred_np = preds_np[b]
                gt_np = labels_np[b]

                pred_left = pts_np[pred_np == id_left]
                pred_right = pts_np[pred_np == id_right]

                gt_left = pts_np[gt_np == id_left]
                gt_right = pts_np[gt_np == id_right]

                pred_left = order_points_pca(pred_left)
                pred_right = order_points_pca(pred_right)
                gt_left = order_points_pca(gt_left)
                gt_right = order_points_pca(gt_right)

                chamfers_left.append(chamfer_distance(pred_left, gt_left))
                chamfers_right.append(chamfer_distance(pred_right, gt_right))
                hausdorff_left.append(hausdorff_distance(pred_left, gt_left))
                hausdorff_right.append(hausdorff_distance(pred_right, gt_right))

    if len(all_preds) == 0:
        raise RuntimeError("No validation data passed (empty dataset?)")

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    accuracy = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average=None, zero_division=0)
    macro_f1 = float(np.mean(f1))

    results = {
        "accuracy": float(accuracy),
        "macro_f1": macro_f1,
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "f1_per_class": f1.tolist(),
        "chamfer_left": float(np.mean(chamfers_left)),
        "chamfer_right": float(np.mean(chamfers_right)),
        "hausdorff_left": float(np.mean(hausdorff_left)),
        "hausdorff_right": float(np.mean(hausdorff_right)),
    }
    return results

