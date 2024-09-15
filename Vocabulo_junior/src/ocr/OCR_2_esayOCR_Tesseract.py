"""
This script performs Optical Character Recognition (OCR) on images using both Tesseract and EasyOCR.
It processes images from a specified input folder, performs OCR, and saves the results and visualizations
of detected text boxes in specified output folders.

The script includes functions to:
- Load images in various formats.
- Resize images to a maximum dimension.
- Perform OCR using Tesseract.
- Perform OCR using EasyOCR.
- Draw bounding boxes around detected text.
- Process images and save the results.

Author: Marianne Arrué
Date: 25/08/24
"""

import os
import cv2
import numpy as np
import pytesseract
import easyocr
from PIL import Image
import pillow_heif
import matplotlib.pyplot as plt

# Configuration
MAX_IMAGE_SIZE = 1600  # Maximum size (width or height) in pixels
INPUT_FOLDER = "Images"  # Folder containing the images
OUTPUT_FOLDER = "Results_1"  # Folder for the results
VISUALIZATION_FOLDER = "Visualizations_1"  # Folder for visualizations

# Create output folders if they do not exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(VISUALIZATION_FOLDER, exist_ok=True)

# Initialize EasyOCR
reader = easyocr.Reader(['fr'])


def load_image(image_path):
    """
    Load an image, regardless of its format.

    Parameters:
    image_path (str): Path to the image file.

    Returns:
    np.ndarray: Loaded image as a NumPy array.
    """
    if image_path.lower().endswith('.heic'):
        heif_file = pillow_heif.read_heif(image_path)
        image = Image.frombytes(
            heif_file.mode,
            heif_file.size,
            heif_file.data,
            "raw",
            heif_file.mode,
            heif_file.stride,
        )
        image = np.array(image)
    else:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def resize_image(image):
    """
    Resize the image if it exceeds the maximum size.

    Parameters:
    image (np.ndarray): Image to be resized.

    Returns:
    np.ndarray: Resized image.
    """
    height, width = image.shape[:2]
    if max(height, width) > MAX_IMAGE_SIZE:
        scale = MAX_IMAGE_SIZE / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height))
    return image


def perform_tesseract_ocr(image):
    """
    Perform OCR using Tesseract and return the text and bounding boxes.

    Parameters:
    image (np.ndarray): Image on which to perform OCR.

    Returns:
    tuple: Detected text and bounding boxes.
    """
    text = pytesseract.image_to_string(image, lang='fra')
    boxes = pytesseract.image_to_boxes(image, lang='fra')
    return text, boxes


def perform_easyocr_ocr(image):
    """
    Perform OCR using EasyOCR and return the text and bounding boxes.

    Parameters:
    image (np.ndarray): Image on which to perform OCR.

    Returns:
    tuple: Detected text and bounding boxes.
    """
    results = reader.readtext(image)
    text = " ".join([result[1] for result in results])
    return text, results


def draw_boxes(image, tesseract_boxes, easyocr_results):
    """
     Draw bounding boxes around detected text by Tesseract and EasyOCR.

     Parameters:
     image (np.ndarray): Original image.
     tesseract_boxes (str): Bounding boxes from Tesseract.
     easyocr_results (list): Bounding boxes from EasyOCR.

     Returns:
     matplotlib.figure.Figure: Figure with drawn bounding boxes.
     """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Tesseract
    ax1.imshow(image)
    height = image.shape[0]
    for b in tesseract_boxes.splitlines():
        b = b.split()
        ax1.add_patch(plt.Rectangle((int(b[1]), height - int(b[2])),
                                    int(b[3]) - int(b[1]),
                                    -(int(b[4]) - int(b[2])),
                                    fill=False, color='red'))
    ax1.set_title('Tesseract OCR')
    ax1.axis('off')

    # EasyOCR
    ax2.imshow(image)
    for (bbox, text, prob) in easyocr_results:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))
        ax2.add_patch(plt.Rectangle(top_left,
                                    bottom_right[0] - top_left[0],
                                    bottom_right[1] - top_left[1],
                                    fill=False, color='blue'))
    ax2.set_title('EasyOCR')
    ax2.axis('off')

    plt.tight_layout()
    return fig


def process_image(image_path):
    """
    Process an image and perform OCR with Tesseract and EasyOCR.

    Parameters:
    image_path (str): Path to the image file.

    Returns:
    tuple: Detected text from Tesseract, detected text from EasyOCR, and figure with drawn bounding boxes.
    """
    print(f"Processing: {image_path}")

    # Load and resize the image
    image = load_image(image_path)
    image = resize_image(image)

    # Perform OCR
    tesseract_text, tesseract_boxes = perform_tesseract_ocr(image)
    easyocr_text, easyocr_results = perform_easyocr_ocr(image)

    # Draw bounding boxes
    fig = draw_boxes(image, tesseract_boxes, easyocr_results)

    return tesseract_text, easyocr_text, fig


def save_results(filename, tesseract_text, easyocr_text, fig):
    """
    Save the OCR results to text files and the image with bounding boxes.

    Parameters:
    filename (str): Original image filename.
    tesseract_text (str): Detected text from Tesseract.
    easyocr_text (str): Detected text from EasyOCR.
    fig (matplotlib.figure.Figure): Figure with drawn bounding boxes.

    Returns:
    None
    """
    base_name = os.path.splitext(filename)[0]

    with open(os.path.join(OUTPUT_FOLDER, f"{base_name}_tesseract2.txt"), "w", encoding="utf-8") as f:
        f.write(tesseract_text)

    with open(os.path.join(OUTPUT_FOLDER, f"{base_name}_easyocr2.txt"), "w", encoding="utf-8") as f:
        f.write(easyocr_text)

    fig.savefig(os.path.join(VISUALIZATION_FOLDER, f"{base_name}_boxes.png"))
    plt.close(fig)


if __name__ == "__main__":
    for filename in os.listdir(INPUT_FOLDER):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.heic')):
            image_path = os.path.join(INPUT_FOLDER, filename)
            tesseract_text, easyocr_text, fig = process_image(image_path)
            save_results(filename, tesseract_text, easyocr_text, fig)

            print(f"Processed: {filename}")
            print("Tesseract Result:")
            print(tesseract_text[:200] + "..." if len(tesseract_text) > 200 else tesseract_text)
            print("\nEasyOCR Result:")
            print(easyocr_text[:200] + "..." if len(easyocr_text) > 200 else easyocr_text)
            print("\n" + "-" * 50 + "\n")

print("Processing complete.")