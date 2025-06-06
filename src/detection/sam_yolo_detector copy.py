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
        # 마스크 이진화 임계값 기본값
        self.bin_th = 0.3

    def run(self, img, use_autocast=True, conf_thresh=0.25, target_classes=None):
        # 1) YOLO 추론
        result = self.detector(img, conf=conf_thresh, classes=target_classes)[0]
        boxes   = result.boxes.xyxy.detach().cpu().numpy()  # (N,4)
        scores  = result.boxes.conf.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy()

        if len(boxes) == 0:
            print("[WARN] YOLO: no detections")
            return None, None

        # 2) 중심점 계산 (각 박스마다 하나씩)
        centers = np.stack([
            (boxes[:,0] + boxes[:,2]) * 0.5,
            (boxes[:,1] + boxes[:,3]) * 0.5
        ], axis=1)  # shape = (N,2)

        # 3) 박스별 리스트로 분할: [[x,y]], [[x,y]], ...
        point_coords = [ centers[i:i+1] for i in range(len(centers)) ]   # list of (1,2)
        point_labels = [ np.array([1], dtype=np.int32) for _ in range(len(centers)) ]  

        # 4) SAM2 세팅
        self.predictor.set_image(img)

        # 5) SAM2 예측
        with torch.inference_mode():
            ctx = torch.autocast(self.device, dtype=torch.bfloat16) if use_autocast else torch.no_grad()
            with ctx:
                masks_list, iou_scores_list, _ = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=boxes,
                    multimask_output=True
                )

        # 6) 후처리 (기존 코드와 동일하게 best mask 선택 & 이진화)
        selected_masks = []
        for idx, (mask_arr, iou_arr) in enumerate(zip(masks_list, iou_scores_list)):
            if mask_arr.ndim == 3:
                best = int(np.argmax(iou_arr))
                chosen = mask_arr[best]
            else:
                chosen = mask_arr
            bin_mask = (chosen > self.bin_th).astype(np.uint8)
            selected_masks.append(bin_mask)

        final_masks = np.stack(selected_masks, axis=0)
        return final_masks, scores