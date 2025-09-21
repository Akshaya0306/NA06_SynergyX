import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tkinter import Tk, filedialog
from fastai.learner import load_learner
from fastai.vision.all import PILImage

# Load trained model
learn = load_learner("leaf_disease_model.pkl")

def predict_disease(img_path):
    img = PILImage.create(img_path)
    pred, _, probs = learn.predict(img)
    prob = float(probs.max())
    
    # Example recommendations
    recommendations = {
        "Healthy": "No action needed. Keep monitoring the crop.",
        "Rust": "Apply fungicide such as Mancozeb or Chlorothalonil.",
        "Blight": "Use copper-based fungicides. Remove infected leaves.",
        "Mildew": "Ensure proper air circulation. Apply sulfur spray."
    }
    recommendation = recommendations.get(str(pred), "Consult an expert for guidance.")
    return str(pred), prob, recommendation

def upload_image():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Leaf Image",
        filetypes=[("Image files", "*.jpg *.JPG *.png *.PNG")]
    )
    root.destroy()

    if file_path:
        pred, prob, recommendation = predict_disease(file_path)
        print(f"\n[UPLOAD] Image: {Path(file_path).name}")
        print(f"Disease Detected: {pred}")
        print(f"Probability: {prob:.4f}")
        print(f"Recommendation: {recommendation}")
    else:
        print("No file selected.")

def scan_leaf():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Cannot access webcam. Please check camera.")
        return

    print("Press 's' to scan leaf, 'q' to quit scanning.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame.")
            break

        # Instead of cv2.imshow -> show frame with matplotlib
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.pause(0.001)  # refresh frame quickly

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            # Save frame temporarily
            img_path = "scanned_leaf.jpg"
            cv2.imwrite(img_path, frame)
            pred, prob, recommendation = predict_disease(img_path)
            print(f"\n[SCAN] Leaf scanned and saved as {img_path}")
            print(f"Disease Detected: {pred}")
            print(f"Probability: {prob:.4f}")
            print(f"Recommendation: {recommendation}")
            break
        elif key == ord("q"):
            break

    cap.release()
    plt.close()

def main():
    while True:
        print("\nSelect an option:")
        print("1. Upload a leaf image")
        print("2. Scan leaf using webcam")
        print("3. Exit")
        choice = input("Enter choice (1/2/3): ")

        if choice == "1":
            upload_image()
        elif choice == "2":
            scan_leaf()
        elif choice == "3":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
