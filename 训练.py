from ultralytics import YOLO
import os

# ---------------------- 核心配置 ----------------------
DATA_YAML = r"C:\Users\小刘\PycharmProjects\ancient_building_system\data\disease_dataset\disease_combined.yaml"
MODEL = "yolov8m.pt"
SAVE_DIR = "runs/detect/train_combined"
EPOCHS = 120
BATCH_SIZE = 8
IMG_SIZE = 800
DEVICE = "0"

# ---------------------- 模型训练 ----------------------
if __name__ == "__main__":
    model = YOLO(MODEL)

    # 删除不支持的val_interval和evolve参数
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        device=DEVICE,
        project=os.path.dirname(SAVE_DIR),
        name=os.path.basename(SAVE_DIR),
        exist_ok=True,
        patience=20,
        lr0=0.01,
        weight_decay=0.0005,
        warmup_epochs=8,
        augment=True,
        mixup=0.15,
        mosaic=1.0,
        val=True,
        workers=0
    )

    best_model_path = os.path.abspath(os.path.join(SAVE_DIR, "weights", "best.pt"))
    print(f"\n🎉 训练完成！")
    print(f"📌 最佳模型绝对路径：{best_model_path}")