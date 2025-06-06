import os
import json
import h5py
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Tuple

import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from evlib.processing.reconstruction import E2Vid # 수정된 E2Vid 클래스 import


def visualize_and_train_sequence(
    h5_path: Path,
    annotation_path: Path,
    output_dir: Path,
    num_event_batch: int = 30000,
    lr: float = 1e-4,
    epochs: int = 5,
):
    """
    1) HDF5 에서 이벤트와 GT 이미지를 읽고
    2) E2VID 모델을 파인튜닝하며
    3) 각 에폭마다 PSNR/SSIM 기록
    4) 최종 체크포인트 및 재구성 이미지 저장
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON 어노테이션 로드 (사용하지 않으면 pass)
    with open(annotation_path, "r") as f:
        annotations = json.load(f)

    # HDF5 데이터
    with h5py.File(h5_path, "r") as f:
        images   = f["images"]       # (T, H, W, C)
        img2ev   = f["img2event"]    # (T+1, )
        events   = f["events"]       # (N_events, 4)
    T = len(images)
    height, width = images[0].shape[:2]

    # E2VID 모델 준비
    e2vid = E2Vid(image_shape=(height, width), use_gpu=torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    e2vid.model.train()
    optimizer = torch.optim.Adam(e2vid.model.parameters(), lr=lr)

    history = {"epoch": [], "psnr": [], "ssim": []}

    # 학습 루프
    for epoch in range(1, epochs + 1):
        total_psnr = 0.0
        total_ssim = 0.0
        for t in tqdm(range(T), desc=f"Epoch {epoch}/{epochs}"):
            start = int(img2ev[t])
            end   = int(img2ev[t+1]) if t+1 < len(img2ev) else len(events)
            ev_slice = np.asarray(events[start:end])
            if len(ev_slice) > num_event_batch:
                ev_slice = ev_slice[-num_event_batch:]

            ev_tensor = e2vid.voxelizer(ev_slice).to(device)
            gt_img = torch.from_numpy(images[t]).permute(2,0,1).unsqueeze(0).float()/255.0
            gt_img = gt_img.to(device)

            optimizer.zero_grad()
            recon_img = e2vid.reconstructor.update_reconstruction(ev_tensor)
            loss = F.mse_loss(recon_img, gt_img)
            loss.backward()
            optimizer.step()

            recon_np = (recon_img.detach().cpu().squeeze().permute(1,2,0).numpy()*255).astype(np.uint8)
            gt_np    = (gt_img.detach().cpu().squeeze().permute(1,2,0).numpy()*255).astype(np.uint8)
            total_psnr += peak_signal_noise_ratio(gt_np, recon_np, data_range=255)
            total_ssim += structural_similarity(gt_np, recon_np, multichannel=True, data_range=255)

        avg_psnr = total_psnr / T
        avg_ssim = total_ssim / T
        history["epoch"].append(epoch)
        history["psnr"].append(avg_psnr)
        history["ssim"].append(avg_ssim)
        print(f"[Epoch {epoch}] PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}")

    # 체크포인트 저장
    ckpt = output_dir / "e2vid_finetuned.pth"
    torch.save(e2vid.model.state_dict(), ckpt)

    # 메트릭 기록
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(history, f, indent=2)

    # 재구성 이미지 시각화 (최초 5 프레임)
    vis_dir = output_dir / "reconstructions"
    vis_dir.mkdir(exist_ok=True)
    e2vid.model.eval()
    with torch.no_grad():
        for t in range(min(5, T)):
            start = int(img2ev[t])
            end   = int(img2ev[t+1]) if t+1 < len(img2ev) else len(events)
            recon = e2vid(np.asarray(events[start:end]))
            gt    = images[t]
            side = np.concatenate([gt, recon], axis=1)
            cv2.imwrite(str(vis_dir / f"frame_{t:03d}.png"), side[..., ::-1])

    print(f"Finished fine-tuning {h5_path.name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'h5_path',
        type=str,
        help='.h5 파일 경로 또는 폴더 경로'
    )
    parser.add_argument(
        'annotation',
        type=str,
        help='annotations JSON 경로'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='저장할 output 폴더 경로'
    )
    parser.add_argument('--batch_size', type=int, default=30000)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--epochs',     type=int,   default=5)
    args = parser.parse_args()

    base_path = Path(args.h5_path)
    ann_path  = Path(args.annotation)
    out_base  = Path(args.output_dir)

    if base_path.is_dir():
        for h5_file in sorted(base_path.glob('*.h5')):
            seq_out = out_base / h5_file.stem
            visualize_and_train_sequence(
                h5_file,
                ann_path,
                seq_out,
                num_event_batch=args.batch_size,
                lr=args.lr,
                epochs=args.epochs,
            )
    else:
        visualize_and_train_sequence(
            base_path,
            ann_path,
            out_base,
            num_event_batch=args.batch_size,
            lr=args.lr,
            epochs=args.epochs,
        )
