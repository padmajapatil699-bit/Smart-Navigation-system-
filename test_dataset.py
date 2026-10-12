from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    model.train(
    data="dataset.yaml",
    epochs=40,
    imgsz=640,
    batch=4,
    device=0,
    workers=0,
    plots=True
)

if __name__ == "__main__":
    main()