"""
This script performs Optical Character Recognition (OCR) on images using Keras-OCR.
It processes images from a specified input folder, performs OCR, and saves the results and visualizations
of detected text boxes in specified output folders.

The script includes functions to:
- Load images in various formats.
- Perform OCR using Keras-OCR.
- Save the recognized text and visualizations of detected text boxes.

Author: Marianne Arrué
Date : 25/08/24
"""

import os
import keras_ocr
import matplotlib.pyplot as plt

# Configuration
INPUT_FOLDER = "Images"  # Folder containing the images
OUTPUT_FOLDER = "Results_Keras"  # Folder for the Keras-OCR results

# Create the output folder if it does not exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize the Keras-OCR pipeline
pipeline = keras_ocr.pipeline.Pipeline()

def process_image(image_path):
    """
    Process an image with Keras-OCR.

    Parameters:
    image_path (str): Path to the image file.

    Returns:
    tuple: Recognized text and predictions (bounding boxes and text).
    """
    print(f"Processing: {image_path}")

    # Load the image
    image = keras_ocr.tools.read(image_path)

    # Perform OCR
    predictions = pipeline.recognize([image])[0]

    # Extract the text
    text = ' '.join([word for (word, box) in predictions])

    return text, predictions

def save_results(filename, text, predictions):
    """
    Save the OCR results to text files and the image with bounding boxes.

    Parameters:
    filename (str): Original image filename.
    text (str): Recognized text.
    predictions (list): Predictions (bounding boxes and text).

    Returns:
    None
    """
    base_name = os.path.splitext(filename)[0]

    # Save the recognized text
    with open(os.path.join(OUTPUT_FOLDER, f"{base_name}_keras.txt"), "w", encoding="utf-8") as f:
        f.write(text)

    # Save the image with text boxes
    image = keras_ocr.tools.read(os.path.join(INPUT_FOLDER, filename))
    fig, ax = plt.subplots(figsize=(10, 20))
    keras_ocr.tools.drawAnnotations(image, predictions, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, f"{base_name}_keras_boxes.png"))
    plt.close()


if __name__ == "__main__":
    for filename in os.listdir(INPUT_FOLDER):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.heic')):
            image_path = os.path.join(INPUT_FOLDER, filename)
            text, predictions = process_image(image_path)
            save_results(filename, text, predictions)

            print(f"Processed: {filename}")
            print("Keras-OCR Result:")
            print(text[:200] + "..." if len(text) > 200 else text)
            print("\n" + "-" * 50 + "\n")

print("Processing complete.")