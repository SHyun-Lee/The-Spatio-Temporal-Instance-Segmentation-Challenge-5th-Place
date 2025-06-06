import ultralytics
import torch
import numpy as np
from sam2.sam2_image_predictor import SAM2ImagePredictor

class Sam2YoloDetector:
    def __init__(self, yolo_path, device='cuda:0') -> None:
        self.device = device
        self.detector = ultralytics.YOLO(yolo_path)
        self.predictor = SAM2ImagePredictor.from_pretrained(
            "facebook/sam2-hiera-large",
            device=device
        )

    def run(self, img, use_autocast=True, conf_thresh=0.25, target_classes=None):
        result = self.detector(
            img,
            conf=conf_thresh,
            classes=target_classes
        )[0]

        boxes   = result.boxes.xyxy.detach().cpu().numpy()
        scores  = result.boxes.conf.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy()

        h, w = img.shape[:2]
        print(f"[DEBUG] Image: H={h}, W={w} | Detections={len(boxes)}")
        for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
            x1, y1, x2, y2 = box
            valid = (0 <= x1 < w) and (0 <= x2 <= w) and (0 <= y1 < h) and (0 <= y2 <= h)
            name = self.detector.names[int(cls)]
            print(f" [{i}] {name:10s} score={score:.3f} box=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}) valid={valid}")
        if len(boxes) == 0:
            print("[WARN] YOLO: no detections")
            return None, None

        self.predictor.set_image(img)

        print("[DEBUG] Calling SAM2.predict()")
        with torch.inference_mode():
            ctx = torch.autocast(self.device, dtype=torch.bfloat16) if use_autocast else torch.no_grad()
            with ctx:
                masks_list, iou_scores_list, _ = self.predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=boxes,
                    multimask_output=None
                )
        print(f"[DEBUG] SAM2 returned masks_list={len(masks_list)}, iou_scores={len(iou_scores_list)}")

        if len(masks_list) == 0:
            print("[WARN] SAM2: no masks returned")
            return None, None


        selected_masks = []
        for i, (mask_arr, iou_arr) in enumerate(zip(masks_list, iou_scores_list)):
            # mask_arr.shape == (k, H, W), iou_arr shape == (k,)
            if mask_arr.ndim == 3:
                best_idx  = int(np.array(iou_arr).argmax())
                chosen    = mask_arr[best_idx]
                best_iou  = float(iou_arr[best_idx])
                print(f"[DEBUG] Obj {i}: {mask_arr.shape[0]} candidates → chosen={best_idx}, IoU={best_iou:.3f}")
            else:
                chosen   = mask_arr
                best_iou = float(iou_arr) if np.ndim(iou_arr)==0 else float(iou_arr[0])
                print(f"[DEBUG] Obj {i}: single candidate, IoU={best_iou:.3f}")

            pix_sum = chosen.sum()
            uniques = np.unique(chosen)
            print(f"        Mask sum={pix_sum:.1f}, uniques={uniques}")
            if pix_sum == 0:
                print(f" [WARN] Empty mask for object {i}")

            bin_mask = (chosen > 0.1).astype(np.uint8)
            selected_masks.append(bin_mask)

        final_masks = np.stack(selected_masks, axis=0)  # shape = (num_boxes, H, W)
        return final_masks, scores
