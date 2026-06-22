import argparse
import os
from ultralytics import YOLO
import cv2
import tkinter as tk

def test_images(model_path, image_dir, conf=0.5):
    # Get screen dimensions using tkinter
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()

    # Load the model
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return

    model = YOLO(model_path)

    # Check if directory exists
    if not os.path.exists(image_dir):
        print(f"Error: Directory '{image_dir}' does not exist.")
        return

    # Supported image extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    # Get list of images
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(valid_extensions)]

    if not image_files:
        print(f"No images found in '{image_dir}'.")
        return

    print(f"Found {len(image_files)} images in '{image_dir}'. Starting inference...")

    # Create an output directory
    output_dir = os.path.join(image_dir, "inference_results")
    os.makedirs(output_dir, exist_ok=True)

    for img_name in image_files:
        img_path = os.path.join(image_dir, img_name)
        print(f"Processing: {img_name}")

        # Run inference
        results = model.predict(source=img_path, save=False, conf=conf)

        # Process results
        for result in results:
            # Plot the results on the image (original size)
            annotated_frame = result.plot()

            # Save the annotated image (original size)
            save_path = os.path.join(output_dir, f"res_{img_name}")
            cv2.imwrite(save_path, annotated_frame)

            # --- Prepare Display Frame (Half Screen + Roll Bar) ---
            h, w = annotated_frame.shape[:2]
            
            # Calculate target display size (half screen width, maintaining aspect ratio)
            display_w = screen_width // 2
            aspect_ratio = h / w
            display_h = int(display_w * aspect_ratio)

            # If the image is too tall for the screen, cap it at half screen height
            if display_h > screen_height:
                display_h = screen_height
                display_w = int(display_h / aspect_ratio)

            display_frame = cv2.resize(annotated_frame, (display_w, display_h))

            # Add Roll Bar (Status Bar) at the bottom of the display frame
            bar_height = 40
            cv2.rectangle(display_frame, (0, display_h - bar_height), (display_w, display_h), (0, 0, 0), -1)
            
            # Add text to roll bar
            status_text = f"File: {img_name} | Conf: {conf}"
            cv2.putText(display_frame, status_text, (10, display_h - 12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            # Display the image on screen
            cv2.imshow("YOLO Detection", display_frame)
            print("Press any key to see next image, or 'q' to quit.")
            if cv2.waitKey(0) & 0xFF == ord('q'):
                print("Exiting...")
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print(f"\nInference complete. Results saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test YOLO model on a folder of images.")
    parser.add_argument("--model", type=str, default="best.pt", help="Path to the trained YOLO model (default: best.pt)")
    parser.add_argument("--dir", type=str, required=True, help="Path to the folder containing images")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (default: 0.5)")

    args = parser.parse_args()

    test_images(args.model, args.dir, args.conf)
