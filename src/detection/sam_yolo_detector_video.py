import os
import cv2
import tempfile
import numpy as np
import torch
import ultralytics
from sam2.sam2_video_predictor import SAM2VideoPredictor
from typing import List, Tuple
import matplotlib.pyplot as plt
from transformers import Mask2FormerModel


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    # box: [x1, y1, x2, y2]
    xa1, ya1, xa2, ya2 = box1
    xb1, yb1, xb2, yb2 = box2
    xi1, yi1 = max(xa1, xb1), max(ya1, yb1)
    xi2, yi2 = min(xa2, xb2), min(ya2, yb2)
    inter_w, inter_h = max(0, xi2 - xi1), max(0, yi2 - yi1)
    inter = inter_w * inter_h
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class Sam2YoloVideoDetector:
    def __init__(
        self,
        yolo_path: str,
        device: str = 'cuda:0',
        frame_size: Tuple[int, int] = (720, 1280),
        save_dir: str = '/home/coraldl/EV/MouseSIS/sam2',
        min_frame_occurrence: int = 5
    ) -> None:
        self.detector = ultralytics.YOLO(yolo_path)
        self.predictor = SAM2VideoPredictor.from_pretrained(
            "facebook/sam2-hiera-large"
        ).to(device)
        self.device = device
        self.height, self.width = frame_size
        self.save_dir = save_dir
        self.min_frame_occurrence = min_frame_occurrence
        os.makedirs(self.save_dir, exist_ok=True)

    def run(
        self,
        video_frames: List[np.ndarray],
        display: bool = False
    ) -> Tuple[List[List[np.ndarray]], np.ndarray]:
        # 1) 프레임별 YOLO 박스 저장
        boxes_per_frame, scores_per_frame = [], []
        for frame in video_frames:
            img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim < 3 else frame
            res = self.detector(img)[0]
            boxes_per_frame.append(res.boxes.xyxy.cpu().numpy())
            scores_per_frame.append(res.boxes.conf.cpu().numpy())

        # 2) 대표 프레임(best_idx) 선택
        best_idx, max_count, best_score = 0, -1, -1.0
        sel_boxes, sel_scores = None, None
        for idx, (boxes, scores) in enumerate(zip(boxes_per_frame, scores_per_frame)):
            cnt, score_sum = boxes.shape[0], float(scores.sum())
            if cnt > max_count or (cnt == max_count and score_sum > best_score):
                best_idx, max_count, best_score = idx, cnt, score_sum
                sel_boxes, sel_scores = boxes.copy(), scores.copy()
        print(f"[DEBUG] YOLO selected best frame {best_idx} with {max_count} boxes")
        if sel_boxes is None or sel_boxes.size == 0:
            return [], np.array([])

        # 3) 추가 조건: 각 sel_box의 등장 횟수 계산 후 최대값 확인
        #    max_count_occurrence < min_frame_occurrence 이면 threshold=-1 사용
        counts = []
        for box in sel_boxes:
            count = sum(
                1 for j, other_boxes in enumerate(boxes_per_frame)
                if j != best_idx and any(calculate_iou(box, ob) > 0.5 for ob in other_boxes)
            )
            counts.append(count)
        max_count_occurrence = max(counts) if counts else -1
        if max_count_occurrence < self.min_frame_occurrence:
            threshold = -1
            print(f"[DEBUG] max occurrence {max_count_occurrence} < {self.min_frame_occurrence}, set threshold={threshold}")
        else:
            threshold = self.min_frame_occurrence
            print(f"[DEBUG] Using threshold={threshold} (min occurrence requirement)")
        # 필터링: count >= threshold (threshold=-1이면 모두 통과)
        filtered_boxes = []
        filtered_scores = []
        for box, score, count in zip(sel_boxes, sel_scores, counts):
            if count >= threshold:
                filtered_boxes.append(box)
                filtered_scores.append(score)
        sel_boxes = np.array(filtered_boxes)
        sel_scores = np.array(filtered_scores)
        print(f"[DEBUG] Filtered to {len(sel_boxes)} boxes with occurrence >= {threshold}")

        # 4) best_idx 재설정: 필터링된 박스 기준으로 가장 많이 등장한 프레임 선택
        if sel_boxes.size > 0:
            # 각 프레임에서 필터된 박스가 몇 번 등장했는지 계산
            occ_counts = []
            for j, other_boxes in enumerate(boxes_per_frame):
                count = sum(
                    1 for box in sel_boxes
                    if any(calculate_iou(box, ob) > 0.5 for ob in other_boxes)
                )
                occ_counts.append(count)
            new_best_idx = int(np.argmax(occ_counts))
            print(f"[DEBUG] Recomputed best_idx from {best_idx} to {new_best_idx} based on filtered boxes occurrence counts")
            best_idx = new_best_idx
        else:
            print(f"[WARN] No filtered boxes to recompute best_idx; keeping original best_idx={best_idx}")

        # 5) 프레임 재배열 (best_idx부터)
        num_frames = len(video_frames)
        order = list(range(best_idx, num_frames)) + list(range(0, best_idx))
        num_frames = len(video_frames)
        order = list(range(best_idx, num_frames)) + list(range(0, best_idx))

        # 5) reordered 프레임을 임시 폴더에 저장하고 SAM2 초기화
        with tempfile.TemporaryDirectory() as tmpdir:
            for new_i, orig_i in enumerate(order):
                frame = video_frames[orig_i]
                img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim < 3 else frame
                cv2.imwrite(os.path.join(tmpdir, f"{new_i:06d}.jpg"), img)

            state = self.predictor.init_state(tmpdir)
            if isinstance(state, tuple):
                state = state[0]

            # 6) reordered 0번 프레임에 프롬프트 추가
            for oid, box in enumerate(sel_boxes):
                box_t = torch.from_numpy(box).float()
                self.predictor.add_new_points_or_box(state, 0, oid, box=box_t)

            # 7) 전체 프레임에 propagate
            masks_all: List[List[np.ndarray]] = []
            for fi, _, masklets in self.predictor.propagate_in_video(state):
                m = (masklets.cpu().numpy() > 0).astype(np.uint8)
                masks_all.append([m[j] for j in range(m.shape[0])])

        masks_seq = masks_all

        # 8) Visualization & save
        for i, mask_list in enumerate(masks_seq):
            frame_idx = order[i]
            out = video_frames[frame_idx].copy()
            for oid, mask in enumerate(mask_list):
                mask2d = mask if mask.ndim == 2 else (mask.sum(axis=0) > 0).astype(np.uint8)
                if mask2d.sum() == 0:
                    continue
                color = tuple(np.random.randint(0, 256, 3).tolist())
                overlay = np.zeros_like(out)
                for c in range(3):
                    overlay[:, :, c] = mask2d * color[c]
                out = cv2.addWeighted(out, 1.0, overlay, 0.5, 0)
                ys, xs = np.nonzero(mask2d)
                if ys.size and xs.size:
                    cv2.putText(out, str(oid), (int(xs.mean()), int(ys.mean())),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            save_path = os.path.join(self.save_dir, f"mask_seq_{frame_idx:06d}.png")
            cv2.imwrite(save_path, out)
            if display:
                plt.figure(figsize=(6, 4))
                plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.show()

        return masks_seq, sel_scores
