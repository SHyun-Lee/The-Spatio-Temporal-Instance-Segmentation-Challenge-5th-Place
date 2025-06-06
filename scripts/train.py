import argparse
from pathlib import Path
import yaml
import h5py
import cv2
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from pycocotools import mask as mask_utils

# 모델 및 트래커 import
from src.detection import SamYoloDetector  # 또는 SamYoloDetector
from src.tracker import XMemSort


def calculate_bbox_from_mask(mask):
    ys, xs = mask.nonzero()
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


class MouseSISFrameDataset(Dataset):
    """
    각 sequence(.h5)와 GT JSON에서
    프레임 단위로 gray 이미지와 GT boxes/labels를 반환합니다.
    """
    def __init__(self, data_root: Path, split: str, gt_folder: Path):
        self.split_dir = data_root / 'top' / split
        self.seq_paths = sorted(self.split_dir.glob('*.h5'))
        self.samples = []  # (seq_path, frame_idx)
        for seq in self.seq_paths:
            with h5py.File(seq, 'r') as f:
                n = len(f['images'])
            for i in range(n):
                self.samples.append((seq, i))
        self.gt_data = {f.stem: json.load(open(f, 'r'))
                        for f in gt_folder.glob('*.json')}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_path, frame_idx = self.samples[idx]
        seq_id = seq_path.stem.replace('seq', '')
        with h5py.File(seq_path, 'r') as f:
            img = f['images'][frame_idx]  # H×W×3 BGR
        # GT masks → boxes
        entries = self.gt_data.get(seq_id, [])
        boxes, labels = [], []
        for e in entries:
            rle = e['segmentations'][frame_idx]
            if rle is None: continue
            mask = mask_utils.decode(rle).astype(bool)
            boxes.append(calculate_bbox_from_mask(mask))
            labels.append(e['category_id'])
        # tensor 변환
        img_t = torch.from_numpy(img[..., ::-1]).permute(2,0,1).float() / 255.0
        target = {'boxes': torch.tensor(boxes, dtype=torch.float32),
                  'labels': torch.tensor(labels, dtype=torch.int64)}
        return img_t, target


def collate_fn(batch):
    imgs, targets = zip(*batch)
    return torch.stack(imgs, 0), list(targets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default='configs/train/detector.yaml')
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, 'r'))

    device = torch.device(cfg['common']['device'])
    data_root = Path(cfg['common']['data_root'])
    gt_folder = Path(cfg['common']['gt_folder'])
    split = cfg['common']['train_split']

    # DataLoader
    ds = MouseSISFrameDataset(data_root, split, gt_folder)
    dl = DataLoader(ds,
                    batch_size=cfg['train']['batch_size'],
                    shuffle=True,
                    num_workers=cfg['train']['num_workers'],
                    collate_fn=collate_fn)

    # 모델 및 트래커 인스턴스화
    model = SamYoloDetector(cfg['model']['yolox']).to(device)
    tracker = XMemSort(**cfg['tracker'], device=device).to(device)
    model.train()
    tracker.train()

    # optimizer에 model과 tracker 파라미터 함께 포함
    optimizer = AdamW(
        list(model.parameters()) + list(tracker.parameters()),
        lr=cfg['train']['lr'],
        weight_decay=cfg['train']['weight_decay']
    )
    scheduler = StepLR(optimizer, step_size=cfg['train']['lr_step'], gamma=cfg['train']['lr_gamma'])

    global_step = 0
    max_iters = cfg['train'].get('max_iterations', len(dl) * cfg['train']['epochs'])
    ckpt_interval = cfg['train'].get('checkpoint_iter', 1000)

    # iteration 기준 학습 루프
    while global_step < max_iters:
        for imgs, targets in dl:
            if global_step >= max_iters:
                break
            imgs = imgs.to(device)

            # 모델 내부에서 detection + tracking loss 반환
            loss = model(imgs, targets=targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            global_step += 1
            if global_step % ckpt_interval == 0 or global_step == max_iters:
                ckpt_dir = Path(cfg['common']['output_dir']) / 'checkpoints'
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'step': global_step,
                    'model_state': model.state_dict(),
                    'tracker_state': tracker.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict()
                }, ckpt_dir / f"iter_{global_step:06d}.pth")
                print(f"Checkpoint saved at iteration {global_step}")

    print("Training finished.")

if __name__ == '__main__':
    main()
