from ultralytics import YOLO

model = YOLO("/models/yolo_e2vid.pt")  


hyp = {
    'lr0': 0.001,  
    'lrf': 0.1,         
    'momentum': 0.937,
    'weight_decay': 0.0005,
}

train_results = model.train(
    data="/configs/yolo/yolo_mouse.yaml",
    epochs=300,
    imgsz=(1280, 720), 
    batch=32,           
    device=[0,1],
    task="detect",       
    resume=False,
    rect=True,           
    project="runs/fine_tune",
    name="yolo11n_ave",
    exist_ok=True
)

val_metrics = model.val(data="/configs/yolo/yolo_mouse.yaml")


img_path = "/data/test_kl/seq01/e2vid/00000032.png"
results = model(img_path, imgsz=640, conf=0.25)  
results[0].show() 

onnx_path = model.export(format="onnx")
print(f"ONNX model saved to: {onnx_path}")
