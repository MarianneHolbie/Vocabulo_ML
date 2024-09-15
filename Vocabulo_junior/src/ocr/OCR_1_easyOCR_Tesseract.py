"""
OCR_1_easyOCR_Tesseract.py

This script performs Optical Character Recognition (OCR) on images using EasyOCR and Tesseract OCR. It includes functions for image preprocessing, skew correction, and text extraction.

Usage:
    python OCR_1_easyOCR_Tesseract.py

Prerequisites:
    - Install necessary packages: easyocr, opencv-python, numpy, pytesseract, pillow, pillow-heif, torch
    - Ensure Tesseract is installed and configured correctly

Author: Marianne Arrué
Date: 20/08/24
"""
import easyocr
import cv2
import numpy as np
import os
import pytesseract
from PIL import Image
import pillow_heif
import logging
from typing import Tuple
import torch

logging.basicConfig(level=logging.DEBUG)


def compute_skew(image: np.ndarray) -> float:
    """
    Computes the skew angle of the image.

    Parameters:
    image (np.ndarray): The input image

    Returns:
    float: The skew angle in degrees
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1)
        angles.append(angle)

    median_angle = np.median(angles)
    return np.degrees(median_angle)


def deskew(image: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Deskews the image and returns the deskewed image and the rotation angle.

    Parameters:
    image (np.ndarray): The input image

    Returns:
    Tuple[np.ndarray, float]: The deskewed image and the rotation angle
    """
    angle = compute_skew(image)
    if abs(angle) < 0.1:  # Si l'angle est très petit, ne pas rotation
        return image, 0
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle


def preprocess_image(image):
    """
   Preprocesses the image by deskewing and converting it to grayscale.

   Parameters:
   image (np.ndarray): The input image

   Returns:
   np.ndarray: The preprocessed grayscale image
   """
    logging.debug("Preprocessing image")

    cv2.imwrite("original_image.png", image)
    logging.info("Saved original image as 'original_image.png'")

    deskewed, angle = deskew(image)
    cv2.imwrite("deskewed_image.png", deskewed)
    logging.info(f"Saved deskewed image as 'deskewed_image.png'. Rotation angle: {angle:.2f} degrees")

    gray = cv2.cvtColor(deskewed, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("grayscale_image.png", gray)
    logging.info("Saved grayscale image as 'grayscale_image.png'")

    return gray


def load_image(image_path):
    """
    Loads an image from the given path.

    Parameters:
    image_path (str): The path to the image file

    Returns:
    np.ndarray: The loaded image
    """
    logging.debug(f"Loading image: {image_path}")
    if image_path.lower().endswith('.heic'):
        heif_file = pillow_heif.read_heif(image_path)
        image = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw",
                                heif_file.mode, heif_file.stride)
    else:
        image = Image.open(image_path)
    return np.array(image)


def perform_ocr_easyocr(image, reader):
    """
    Performs OCR on the image using EasyOCR.

    Parameters:
    image (np.ndarray): The input image
    reader (easyocr.Reader): The EasyOCR reader instance

    Returns:
    list: The OCR results from EasyOCR
    """
    logging.debug("Performing EasyOCR")
    results = reader.readtext(image)
    logging.debug(f"EasyOCR found {len(results)} text regions")
    return results


def perform_ocr_tesseract(image):
    """
    Performs OCR on the image using Tesseract.

    Parameters:
    image (np.ndarray): The input image

    Returns:
    str: The OCR result from Tesseract
    """
    logging.debug("Performing Tesseract OCR")
    try:
        text = pytesseract.image_to_string(image, lang='fra')
        logging.debug(f"Tesseract OCR result length: {len(text)}")
    except Exception as e:
        logging.error(f"Tesseract OCR failed: {str(e)}")
        text = ""
    return text


def postprocess_ocr_results_easyocr(results):
    """
    Postprocesses the OCR results from EasyOCR.

    Parameters:
    results (list): The OCR results from EasyOCR

    Returns:
    str: The concatenated text from the OCR results
    """
    return " ".join([detection[1] for detection in results])


def process_single_image(image_path, easyocr_reader):
    """
    Processes a single image and performs OCR using both EasyOCR and Tesseract.

    Parameters:
    image_path (str): The path to the image file
    easyocr_reader (easyocr.Reader): The EasyOCR reader instance

    Returns:
    Tuple[str, str]: The OCR results from EasyOCR and Tesseract
    """
    logging.info(f"Processing single image: {image_path}")
    image = load_image(image_path)

    preprocessed_image = preprocess_image(image)

    ocr_results_easyocr = perform_ocr_easyocr(preprocessed_image, easyocr_reader)
    page_text_easyocr = postprocess_ocr_results_easyocr(ocr_results_easyocr)

    page_text_tesseract = perform_ocr_tesseract(preprocessed_image)

    logging.info(f"EasyOCR result: {page_text_easyocr}")
    logging.info(f"Tesseract result: {page_text_tesseract}")

    return page_text_easyocr, page_text_tesseract


def process_images_in_folder(folder_path):
    """
    Processes all images in a folder and performs OCR using both EasyOCR and Tesseract.

    Parameters:
    folder_path (str): The path to the folder containing images

    Returns:
    Tuple[str, str]: The concatenated OCR results from EasyOCR and Tesseract for all images
    """
    easyocr_reader = easyocr.Reader(['fr'])

    all_text_easyocr = ""
    all_text_tesseract = ""

    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.heic')):
            image_path = os.path.join(folder_path, filename)
            logging.info(f"Processing {filename}...")

            try:
                page_text_easyocr, page_text_tesseract = process_single_image(image_path, easyocr_reader)
                all_text_easyocr += page_text_easyocr + "\n\n"
                all_text_tesseract += page_text_tesseract + "\n\n"

                logging.info(f"Successfully processed {filename}")
            except Exception as e:
                logging.error(f"Error processing {filename}: {str(e)}")

    return all_text_easyocr, all_text_tesseract


if __name__ == "__main__":
    folder_path = "./Images"

    extracted_text_easyocr, extracted_text_tesseract = process_images_in_folder(folder_path)

    print("EasyOCR results:")
    print(extracted_text_easyocr)
    print("\nTesseract results:")
    print(extracted_text_tesseract)

    with open("easyocr_results.txt", "w", encoding="utf-8") as f:
        f.write(extracted_text_easyocr)
    with open("tesseract_results.txt", "w", encoding="utf-8") as f:
        f.write(extracted_text_tesseract)

    if os.listdir(folder_path):
        single_image_path = os.path.join(folder_path, os.listdir(folder_path)[0])
        gpu = torch.cuda.is_available()
        easyocr_reader = easyocr.Reader(['fr'], gpu=gpu)
        process_single_image(single_image_path, easyocr_reader)