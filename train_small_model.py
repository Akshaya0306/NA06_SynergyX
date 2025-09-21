from fastai.vision.all import *
from pathlib import Path

def main():
    # Path to your small dataset
    path = Path(r"C:\Users\aksha\Downloads\small_leaf_dataset")  # <-- update if needed

    # Check if path exists
    if not path.exists():
        print(f"Error: Path {path} does not exist!")
        return

    # List subfolders and count images
    print("Checking dataset folders and image counts...")
    for folder in path.ls():
        if folder.is_dir():
            images = list(folder.glob('*'))
            print(f"{folder.name}: {len(images)} images")

    # Create DataLoaders
    dls = ImageDataLoaders.from_folder(
        path,
        valid_pct=0.2,          # 20% for validation
        seed=42,
        item_tfms=Resize(224),  # Resize images to 224x224
        bs=16,                  # batch size
        num_workers=0           # required for Windows
    )

    # Create a simple learner with ResNet18
    learn = vision_learner(dls, resnet18, metrics=accuracy)

    # Fine-tune for 2 epochs (fast for small dataset)
    learn.fine_tune(2)

    # Export the trained model
    model_path = Path(r"C:\Users\aksha\OneDrive\Desktop\agricare\leaf_disease_model.pkl")
    learn.export(model_path)
    print(f"Model saved successfully at {model_path}")

if __name__ == "__main__":
    main()
