import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import argparse
import json

def rle_to_mask(rle, height, width):
    '''Convert a run-length encoded representation of the mask to a binary mask.'''
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height*width, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape((width, height)).T

def make_overlay_image(image, events):
    y, x, p = events[:, 0], events[:, 1], events[:, 3]
    image = np.copy(image)
    image[y[p == 0], x[p == 0]] = (255, 0, 0)  # Red for negative events
    image[y[p == 1], x[p == 1]] = (0, 0, 255)  # Blue for positive events
    return image

def create_color_map(n_instances):
    """Create a color map for instances with distinct colors."""
    colors = [
        (239, 65, 53),    # Red
        (46, 134, 193),   # Blue
        (39, 174, 96),    # Green
        (142, 68, 173),   # Purple
        (241, 196, 15),   # Yellow
        (230, 126, 34),   # Orange
        (52, 152, 219),   # Light Blue
        (231, 76, 60),    # Light Red
        (46, 204, 113),   # Light Green
        (155, 89, 182),   # Light Purple
        (26, 188, 156),   # Turquoise
        (251, 206, 25),   # Bright Yellow
    ]
    while len(colors) < n_instances:
        new_color = tuple(np.random.randint(50, 240, 3))
        colors.append(new_color)
    return colors

def visualize_sequence(h5_path, annotation_path, output_dir):
    num_event_batch = 30000
    h5_path = Path(h5_path)
    output_dir = Path(output_dir)
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    with open(annotation_path, 'r') as f:
        annotations = json.load(f)

    video_id = h5_path.stem.split('_')[-1][-2:]  # Assuming format seq_XX.hdf5
    video_annotations = [ann for ann in annotations['annotations'] 
                        if ann['video_id'] == video_id.zfill(2)]
    
    with h5py.File(h5_path, 'r') as f:
        images = f['images']
        img2ev = f['img2event']
        height, width = images[0].shape[:2]

        colors = create_color_map(len(video_annotations))
        instance_colors = {ann['id']: colors[i] for i, ann in enumerate(video_annotations)}
        
        for i, (ev_index, img) in tqdm(enumerate(zip(img2ev, images)), total=len(img2ev)):
            start_index = int(max(0, ev_index - 0.5 * num_event_batch))
            end_index = int(min(ev_index + 0.5 * num_event_batch, len(f['y'])))
            
            events = np.zeros((end_index - start_index, 4), dtype=int)
            events[:, 0] = f["y"][start_index:end_index]
            events[:, 1] = f["x"][start_index:end_index]
            events[:, 2] = f["t"][start_index:end_index]
            events[:, 3] = f["p"][start_index:end_index]
            
            events = events[events[:, 0] < height]
            events = events[events[:, 1] < width]
            
            overlay = make_overlay_image(img.copy(), events)
            mask_overlay = np.zeros((height, width, 3), dtype=np.uint8)

            for ann in video_annotations:
                if ann['segmentations'][i]:
                    rle = ann['segmentations'][i]['counts']
                    mask = rle_to_mask(rle, height, width)
                    
                    color = instance_colors[ann['id']]
                    for c in range(3):
                        mask_overlay[..., c][mask == 1] = color[c]


            final_img = mask_overlay

            Image.fromarray(final_img).save(output_dir / f'visualization_{str(i).zfill(5)}.png')
            
            # for ann in video_annotations:
            #     if ann['segmentations'][i]:
            #         rle = ann['segmentations'][i]['counts']
            #         mask = rle_to_mask(rle, height, width)
                    
            #         color = instance_colors[ann['id']]
            #         for c in range(3):
            #             mask_overlay[..., c][mask == 1] = color[c]
            
            # alpha = 0.7
            # mask_overlay = (img.astype(float) * alpha + mask_overlay.astype(float) * (1 - alpha)).astype(np.uint8)
            # final_img = np.concatenate([overlay, mask_overlay], axis=1)
            # Image.fromarray(final_img).save(output_dir / f'visualization_{str(i).zfill(5)}.png')
    
    print('Done!')

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize event data and masks from H5 file')
    parser.add_argument(
        '--h5_path',
        help='Path to the sequence H5 file',
        type=str,
        required=True,
        default='/data/MouseSIS/top/test'
    )
    parser.add_argument(
        '--annotation_path',
        help='Path to the annotation JSON file',
        type=str,
        required=True,
        default='/data/MouseSIS/train_annotations.json'
    )
    parser.add_argument(
        '--output_dir',
        help='Output directory for visualizations',
        type=str,
        default='/data/MouseSIS/Mask_val'
    )
    return parser.parse_args()

def main():
    args = parse_args()
    h5_path = Path(args.h5_path)
    output_dir = Path(args.output_dir)

    if h5_path.is_dir():
        h5_files = sorted(list(h5_path.glob("*.h5")) + list(h5_path.glob("*.hdf5")))
    else:
        h5_files = [h5_path]

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    for file in h5_files:
        print(f"Processing file: {file}")
        file_output_dir = output_dir / file.stem
        if not file_output_dir.exists():
            file_output_dir.mkdir(parents=True)
        visualize_sequence(str(file), args.annotation_path, str(file_output_dir))

if __name__ == '__main__':
    main()