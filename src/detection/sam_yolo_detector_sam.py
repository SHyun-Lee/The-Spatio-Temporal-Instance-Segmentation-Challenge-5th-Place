import ultralytics
from transformers import SamModel, SamProcessor
import torch

from ..utils import suppress_stdout_stderr


class SamYoloDetector:
    def __init__(self, yolo_path, device='cuda:0') -> None:
        self.detector = ultralytics.YOLO(yolo_path)
        self.sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
        self.sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
        self.device = device

    def run(self, img):
        with suppress_stdout_stderr():
            result = self.detector(img)[0]
        
        boxes = result.boxes.xyxy.detach().cpu().numpy()  # x1, y1, x2, y2
        scores = result.boxes.conf.detach().cpu().numpy()
        
        if not len(boxes):
            return None, None
            
        boxes_list = [[boxes.tolist()]]
        inputs = self.sam_processor(img.transpose(2, 0, 1), input_boxes=[boxes_list], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.sam_model(**inputs)
            masks = self.sam_processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu()
            )[0]
            
            iou_scores = outputs.iou_scores.cpu()[0]
            num_instances, nb_predictions, height, width = masks.shape
            max_indices = iou_scores.argmax(dim=1, keepdim=True)
            gather_indices = max_indices[..., None, None].expand(-1, 1, height, width)
            selected_masks = torch.gather(masks, 1, gather_indices).squeeze(1)
            
        return selected_masks.cpu().numpy(), scores